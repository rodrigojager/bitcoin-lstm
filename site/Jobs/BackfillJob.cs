using Quartz;
using Microsoft.Extensions.Options;
using System.Net.Http.Json;
using TechChallenge.Models;
using Microsoft.Extensions.Logging;

namespace TechChallenge.Jobs
{
    public class BackfillJob : IJob
    {
        private readonly IHttpClientFactory _http;
        private readonly ApiSettings _cfg;
        private readonly ILogger<BackfillJob> _logger;

        public BackfillJob(IHttpClientFactory http, IOptions<ApiSettings> cfg, ILogger<BackfillJob> logger)
        {
            _http = http;
            _cfg = cfg.Value;
            _logger = logger;
        }

        public async Task Execute(IJobExecutionContext context)
        {
            if (!_cfg.RunBackfillOnStartup) return;

            var client = _http.CreateClient();

            try
            {
                client.Timeout = TimeSpan.FromMinutes(30);
                // 1) Checa cobertura simples via /series
                int days = Math.Min(_cfg.BackfillDays, 90);
                var series = await client.GetFromJsonAsync<SeriesResponse>($"{_cfg.BaseUrl}/series?fallback_days={days}");

                // Aprox: 90d de 5m ~ 25.920 pontos
                int expected = 25920 * days / 90;
                int got = series?.points?.Count ?? 0;
                double ratio = expected > 0 ? (double)got / expected : 0;

                if (ratio < _cfg.ExpectedCoverageRatio)
                {
                    client.Timeout = TimeSpan.FromMinutes(30);
                    await client.PostAsync($"{_cfg.BaseUrl}/init/backfill", null);

                    // Após backfill, garantir que exista modelo e série materializada para os gráficos
                    await client.PostAsync($"{_cfg.BaseUrl}/train?days={days}", null);
                    await client.PostAsync($"{_cfg.BaseUrl}/series/rebuild?days={days}", null);
                    await client.PostAsync($"{_cfg.BaseUrl}/futures/update", null);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "BackfillJob falhou");
            }
        }

        // Tipos p/ desserializar parcialmente o /series
        public class SeriesResponse { public List<PointItem> points { get; set; } = new(); }
        public class PointItem { public RealItem real { get; set; } = default!; }
        public class RealItem { public string time { get; set; } = ""; }
    }
}
