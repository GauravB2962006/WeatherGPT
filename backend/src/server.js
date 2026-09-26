import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import weatherRoutes from './routes/weather.js';
import chatRoutes from './routes/chat.js';

dotenv.config();
const app = express();
app.use(cors());
app.use(express.json());

app.get('/api/health', (_req, res) => res.json({ ok: true, service: 'WeatherGPT backend' }));
app.use('/api/weather', weatherRoutes);
app.use('/api/chat', chatRoutes);

const port = process.env.PORT || 4000;
app.listen(port, () => console.log(`WeatherGPT backend running on http://localhost:${port}`));
