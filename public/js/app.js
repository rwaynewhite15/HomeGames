/* ===== App entry point ===== */

// Close modals on overlay click
document.querySelectorAll('.modal-overlay').forEach(overlay => {
  overlay.addEventListener('click', e => {
    if (e.target === overlay) overlay.classList.remove('open');
  });
});

// Enter key in player name input
document.getElementById('player-name-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') savePlayer();
});

// Init: load players then show lobby
(async () => {
  try {
    await refreshPlayers();
  } catch (e) {
    console.error('Failed to load players on startup:', e);
  }
  showView('lobby');
})();
