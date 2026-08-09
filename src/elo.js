/**
 * ELO rating calculation utilities
 * Standard ELO with K-factor based on rating range
 */

function getKFactor(elo) {
  if (elo < 1400) return 32;
  if (elo < 1800) return 24;
  return 16;
}

function expectedScore(ratingA, ratingB) {
  return 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
}

/**
 * Calculate new ELO ratings after a game
 * @param {number} ratingA
 * @param {number} ratingB
 * @param {string} result - 'A' if A wins, 'B' if B wins, 'draw'
 * @returns {{ newA: number, newB: number, changeA: number, changeB: number }}
 */
function calculateElo(ratingA, ratingB, result) {
  const K = Math.max(getKFactor(ratingA), getKFactor(ratingB));
  const expA = expectedScore(ratingA, ratingB);
  const expB = 1 - expA;

  let scoreA, scoreB;
  if (result === 'A') {
    scoreA = 1; scoreB = 0;
  } else if (result === 'B') {
    scoreA = 0; scoreB = 1;
  } else {
    scoreA = 0.5; scoreB = 0.5;
  }

  const changeA = Math.round(K * (scoreA - expA));
  const changeB = Math.round(K * (scoreB - expB));

  return {
    newA: Math.max(100, ratingA + changeA),
    newB: Math.max(100, ratingB + changeB),
    changeA,
    changeB,
  };
}

module.exports = { calculateElo };
