/* ===== API helpers ===== */
const API = {
  async getPlayers() {
    const r = await fetch('/api/players');
    if (!r.ok) throw new Error('Failed to fetch players');
    return r.json();
  },
  async createPlayer(name, avatar) {
    const r = await fetch('/api/players', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, avatar }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Failed to create player');
    return data;
  },
  async updatePlayer(id, name, avatar) {
    const r = await fetch(`/api/players/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, avatar }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Failed to update player');
    return data;
  },
  async deletePlayer(id) {
    const r = await fetch(`/api/players/${id}`, { method: 'DELETE' });
    if (!r.ok) { const d = await r.json(); throw new Error(d.error || 'Failed to delete'); }
    return r.json();
  },
  async recordResult(game_type, player_ids, winner_id) {
    const r = await fetch('/api/games/result', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_type, player_ids, winner_id }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Failed to record result');
    return data;
  },
  async getHistory(limit = 50) {
    const r = await fetch(`/api/games/history?limit=${limit}`);
    if (!r.ok) throw new Error('Failed to fetch history');
    return r.json();
  },
};
