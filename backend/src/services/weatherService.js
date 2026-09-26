const timeout = 10000;

async function fetchJson(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`Weather provider returned ${response.status}`);
    return await response.json();
  } finally { clearTimeout(timer); }
}

export async function getForecastByCity(city) {
  const geo = await fetchJson(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1&language=en&format=json`);
  if (!geo.results?.length) throw new Error(`Location not found: ${city}`);
  const place = geo.results[0];
  const url = new URL('https://api.open-meteo.com/v1/forecast');
  url.searchParams.set('latitude', place.latitude);
  url.searchParams.set('longitude', place.longitude);
  url.searchParams.set('current', 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,surface_pressure');
  url.searchParams.set('hourly', 'temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m,surface_pressure');
  url.searchParams.set('daily', 'temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code,wind_speed_10m_max');
  url.searchParams.set('forecast_days', '7');
  url.searchParams.set('timezone', 'auto');
  const data = await fetchJson(url.toString());
  return { location: { name: place.name, country: place.country, latitude: place.latitude, longitude: place.longitude }, ...data };
}
