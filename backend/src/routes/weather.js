import { Router } from 'express';
import { getForecastByCity } from '../services/weatherService.js';

const router = Router();
router.get('/forecast', async (req, res) => {
  try {
    const city = String(req.query.city || '').trim();
    if (!city) return res.status(400).json({ error: 'city is required' });
    res.json(await getForecastByCity(city));
  } catch (error) {
    res.status(502).json({ error: error.message || 'Weather service failed' });
  }
});
export default router;
