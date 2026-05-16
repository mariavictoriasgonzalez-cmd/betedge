// ════════════════════════════════════════════════════════════════════════════
// BetEdge · Modelo Poisson ponderado por forma reciente
// ════════════════════════════════════════════════════════════════════════════
// Unifica las tres implementaciones que estaban duplicadas en index.html:
//   - getWeightedProbs1x2()  (simulación histórica de 1X2)
//   - wStats2()              (predicción para partidos en vivo)
//   - weightedTeamStats()    (calculadora manual)
//
// API pública (todo bajo window.BetEdgeModel):
//   BetEdgeModel.predict(opts)   → predicción completa de un partido
//   BetEdgeModel.recentForm(...) → forma reciente de un equipo (últimos N)
//   BetEdgeModel.h2h(...)        → estadísticas head-to-head
//   BetEdgeModel.CONFIG          → constantes editables del modelo
//
// Diseño:
//   - Función pura: no toca el DOM ni lee sliders. Le pasas los datos que necesita.
//   - Una sola lambda canónica. Si más adelante migramos a backend, se porta tal cual.
//   - Tres "modos de filtro" para reproducir cada uno de los usos originales.
// ════════════════════════════════════════════════════════════════════════════

(function (global) {
  'use strict';

  // ── Constantes del modelo (mover aquí cualquier "magic number") ─────────────
  const CONFIG = {
    DECAY: 0.92,            // factor de decaimiento por partido (0.92 = ~50% peso a los 8 últimos)
    WINDOW: 15,             // últimos N partidos por equipo para calcular forma
    LEAGUE_HOME_AVG: 1.45,  // goles medios del local en La Liga (histórico)
    LEAGUE_AWAY_AVG: 1.15,  // goles medios del visitante en La Liga
    HOME_ADV: 1.08,         // multiplicador de ventaja de localía
    LAMBDA_MIN: 0.3,        // clip inferior de goles esperados (evita extremos)
    LAMBDA_MAX: 4.5,        // clip superior
    POISSON_K: 9,           // matriz de convolución K×K (9×9 cubre >99.9% prob)
    OU_LINE: 2.5,           // línea Over/Under
    FALLBACK_GF: 1.25,      // si un equipo no tiene historia, valor por defecto
    FALLBACK_GA: 1.15,
    RECENT_SEASONS: ['2022-23','2023-24','2024-25','2025-26'], // ventana "calculadora"
  };

  // ── Helpers internos ────────────────────────────────────────────────────────

  // Probabilidad Poisson: P(X=k | λ=l)
  function poisson(l, k) {
    let r = Math.exp(-l), p = 1;
    for (let i = 1; i <= k; i++) p *= l / i;
    return r * p;
  }

  // Clip a [min, max]
  function clip(v, min, max) { return Math.max(min, Math.min(max, v)); }

  // ── Estadísticas ponderadas de un equipo ───────────────────────────────────
  //
  // historicalData: array de partidos (RAW completo o filtrado por temporada)
  // team:           nombre del equipo
  // isHome:         true → estadísticas como local, false → como visitante
  // formWeight:     [0..1] — 0 = media simple, 1 = solo ponderado por forma
  // dateLimit:      string 'YYYY-MM-DD' opcional — solo cuenta partidos ANTERIORES
  //                 (para evitar data leakage en backtests sobre histórico)
  //
  // Devuelve: { gf, ga, sample } — goles a favor/en contra esperados + tamaño muestra
  function teamStats(historicalData, team, isHome, formWeight, dateLimit) {
    const gKey  = isHome ? 'fthg' : 'ftag';
    const gaKey = isHome ? 'ftag' : 'fthg';

    let matches = historicalData.filter(r =>
      (isHome ? r.home === team : r.away === team) &&
      r[gKey] != null
    );
    if (dateLimit) matches = matches.filter(r => r.date < dateLimit);

    matches = matches
      .sort((a, b) => b.date.localeCompare(a.date))
      .slice(0, CONFIG.WINDOW);

    if (!matches.length) {
      return { gf: CONFIG.FALLBACK_GF, ga: CONFIG.FALLBACK_GA, sample: 0 };
    }

    // Media simple
    const simpleGF = matches.reduce((s, m) => s + (m[gKey]  || 0), 0) / matches.length;
    const simpleGA = matches.reduce((s, m) => s + (m[gaKey] || 0), 0) / matches.length;

    if (formWeight === 0) {
      return { gf: simpleGF, ga: simpleGA, sample: matches.length };
    }

    // Media ponderada por decaimiento exponencial
    let wgf = 0, wga = 0, ws = 0;
    matches.forEach((m, i) => {
      const w = Math.pow(CONFIG.DECAY, i);
      wgf += (m[gKey]  || 0) * w;
      wga += (m[gaKey] || 0) * w;
      ws  += w;
    });
    const weightedGF = wgf / ws;
    const weightedGA = wga / ws;

    // Blend: forma reciente vs histórico
    return {
      gf: weightedGF * formWeight + simpleGF * (1 - formWeight),
      ga: weightedGA * formWeight + simpleGA * (1 - formWeight),
      sample: matches.length,
    };
  }

  // ── Cálculo de lambdas (goles esperados) ───────────────────────────────────
  function expectedGoals(hStats, aStats) {
    const lgH = CONFIG.LEAGUE_HOME_AVG;
    const lgA = CONFIG.LEAGUE_AWAY_AVG;
    const lamH = clip(
      (hStats.gf / lgH) * (aStats.ga / lgH) * lgH * CONFIG.HOME_ADV,
      CONFIG.LAMBDA_MIN, CONFIG.LAMBDA_MAX
    );
    const lamA = clip(
      (aStats.gf / lgA) * (hStats.ga / lgA) * lgA,
      CONFIG.LAMBDA_MIN, CONFIG.LAMBDA_MAX
    );
    return { lamH, lamA };
  }

  // ── Convolución de dos Poisson → P(H), P(D), P(A), P(Over), P(Under) ──────
  function convolve(lamH, lamA) {
    const K = CONFIG.POISSON_K;
    let pH = 0, pD = 0, pA = 0;
    let pOver = 0;
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        const p = poisson(lamH, i) * poisson(lamA, j);
        if (i > j)      pH += p;
        else if (i === j) pD += p;
        else            pA += p;
        if (i + j > CONFIG.OU_LINE) pOver += p;
      }
    }
    const total = pH + pD + pA;
    // Normalizar 1X2 (la cola truncada en K introduce micro-error)
    return {
      pH: pH / total,
      pD: pD / total,
      pA: pA / total,
      pOver,
      pUnder: 1 - pOver,
    };
  }

  // ────────────────────────────────────────────────────────────────────────────
  // API PÚBLICA
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * Predice un partido completo.
   *
   * @param {Object} opts
   * @param {string} opts.home                Nombre del equipo local
   * @param {string} opts.away                Nombre del equipo visitante
   * @param {Array}  opts.historicalData      Array de partidos para entrenar (RAW o subset)
   * @param {number} [opts.formWeight=0.5]    [0..1] — peso de la forma reciente
   * @param {string} [opts.dateLimit]         'YYYY-MM-DD' — solo cuenta partidos anteriores
   *                                          (úsalo en backtests para no mirar al futuro)
   *
   * @returns {Object} {
   *   model_h, model_d, model_a,    // probabilidades en porcentaje (suman 100)
   *   prob_over, prob_under,        // O/U 2.5 en porcentaje
   *   exp_goals,                    // total goles esperados (lamH+lamA)
   *   lamH, lamA,                   // goles esperados por equipo
   *   sample_h, sample_a,           // tamaño de muestra usado para cada equipo
   * }
   */
  function predict(opts) {
    const { home, away, historicalData, formWeight = 0.5, dateLimit } = opts;
    if (!home || !away || !historicalData) {
      throw new Error('BetEdgeModel.predict: faltan home/away/historicalData');
    }

    const fw = clip(formWeight, 0, 1);
    const hStats = teamStats(historicalData, home, true,  fw, dateLimit);
    const aStats = teamStats(historicalData, away, false, fw, dateLimit);
    const { lamH, lamA } = expectedGoals(hStats, aStats);
    const probs = convolve(lamH, lamA);

    return {
      model_h:   +(probs.pH * 100).toFixed(1),
      model_d:   +(probs.pD * 100).toFixed(1),
      model_a:   +(probs.pA * 100).toFixed(1),
      prob_over: +(probs.pOver  * 100).toFixed(1),
      prob_under:+(probs.pUnder * 100).toFixed(1),
      exp_goals: +(lamH + lamA).toFixed(2),
      lamH: +lamH.toFixed(3),
      lamA: +lamA.toFixed(3),
      sample_h: hStats.sample,
      sample_a: aStats.sample,
    };
  }

  /**
   * Forma reciente de un equipo (últimos N partidos jugados).
   * No requiere modelo, es estadística pura.
   */
  function recentForm(historicalData, team, n = 5) {
    const matches = historicalData
      .filter(r => (r.home === team || r.away === team) && r.actual)
      .sort((a, b) => b.date.localeCompare(a.date))
      .slice(0, n);

    return matches.map(r => {
      const isHome = r.home === team;
      const res    = r.actual;
      const win    = (isHome && res === 'H') || (!isHome && res === 'A');
      const draw   = res === 'D';
      return {
        win, draw, loss: !win && !draw,
        gf: isHome ? r.fthg : r.ftag,
        ga: isHome ? r.ftag : r.fthg,
        opponent: isHome ? r.away : r.home,
        date: r.date,
      };
    });
  }

  /**
   * Puntuación de forma (0-100): W=3, D=1, L=0 sobre últimos N (normalizado).
   */
  function formScore(form) {
    if (!form.length) return 0;
    const pts    = form.reduce((s, r) => s + (r.win ? 3 : r.draw ? 1 : 0), 0);
    const maxPts = form.length * 3;
    return Math.round((pts / maxPts) * 100);
  }

  /**
   * Tendencia: +N si está mejorando, -N si está empeorando.
   * Compara los 2 más recientes vs los anteriores.
   */
  function formTrend(form) {
    if (form.length < 4) return 0;
    const recent = form.slice(0, 2).reduce((s, r) => s + (r.win ? 3 : r.draw ? 1 : 0), 0) / 2;
    const older  = form.slice(2).reduce((s, r) => s + (r.win ? 3 : r.draw ? 1 : 0), 0) / (form.length - 2);
    return +(recent - older).toFixed(2);
  }

  /**
   * Head-to-head entre dos equipos.
   */
  function h2h(historicalData, home, away, n = 10) {
    const matches = historicalData
      .filter(r =>
        ((r.home === home && r.away === away) || (r.home === away && r.away === home)) &&
        r.actual)
      .sort((a, b) => b.date.localeCompare(a.date))
      .slice(0, n);

    const homeWins = matches.filter(r =>
      (r.home === home && r.actual === 'H') || (r.away === home && r.actual === 'A')
    ).length;
    const awayWins = matches.filter(r =>
      (r.home === away && r.actual === 'H') || (r.away === away && r.actual === 'A')
    ).length;
    const draws    = matches.filter(r => r.actual === 'D').length;
    const avgGoals = matches.length
      ? +(matches.reduce((s, r) => s + (r.fthg + r.ftag), 0) / matches.length).toFixed(2)
      : null;
    const bothScored = matches.filter(r => r.fthg > 0 && r.ftag > 0).length;

    return { matches, homeWins, awayWins, draws, avgGoals, bothScored };
  }

  /**
   * Helper para la calculadora: usa solo las últimas N temporadas si hay
   * suficientes datos, si no usa todo el histórico.
   * Replica la lógica que estaba en weightedTeamStats().
   */
  function selectTrainingData(historicalData, home, away, minMatches = 8) {
    const recent = historicalData.filter(r => CONFIG.RECENT_SEASONS.includes(r.season));
    const hCount = recent.filter(r => r.home === home || r.away === home).length;
    const aCount = recent.filter(r => r.home === away || r.away === away).length;
    return (hCount >= minMatches && aCount >= minMatches) ? recent : historicalData;
  }

  // ── Export ──────────────────────────────────────────────────────────────────
  const API = {
    predict,
    recentForm,
    formScore,
    formTrend,
    h2h,
    selectTrainingData,
    CONFIG,
    // Helpers internos expuestos para tests o debugging
    _internals: { teamStats, expectedGoals, convolve, poisson },
  };

  // Browser
  if (typeof window !== 'undefined') {
    global.BetEdgeModel = API;
  }
  // Node (futuro: backend / tests)
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
  }

})(typeof globalThis !== 'undefined' ? globalThis : this);
