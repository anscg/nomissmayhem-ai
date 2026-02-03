import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
app.use(express.json());
// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// In-memory leaderboard
let leaderboard = [];

// API endpoints
app.get('/api/leaderboard', (req, res) => {
  const sortedLeaderboard = leaderboard.sort((a, b) => b.score - a.score).slice(0, 10);
  res.json(sortedLeaderboard);
});

app.post('/api/leaderboard', (req, res) => {
  const { name, score, time } = req.body;
  leaderboard.push({ name, score, time });
  res.json({ message: 'Score submitted', score: { name, score, time } });
});

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
