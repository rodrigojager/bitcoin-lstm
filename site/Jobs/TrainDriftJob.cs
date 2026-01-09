using Quartz;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Logging;
using System.Net.Http.Json;
using TechChallenge.Models;

namespace TechChallenge.Jobs
{
    /// <summary>
    /// Checagem de drift (Opção C): roda periodicamente e treina somente se:
    /// - passou >= TrainMinHours desde o último treino e
    /// - MAPE(rolling) em futures >= FuturesMapeThreshold
    /// </summary>
    public class TrainDriftJob : IJob
    {
        private readonly IHttpClientFactory _http;
        private readonly ApiSettings _cfg;
        private readonly ILogger<TrainDriftJob> _logger;

        public TrainDriftJob(IHttpClientFactory http, IOptions<ApiSettings> cfg, ILogger<TrainDriftJob> logger)
        {
            _http = http;
            _cfg = cfg.Value;
            _logger = logger;
        }

        public async Task Execute(IJobExecutionContext context)
        {
            var client = _http.CreateClient();
            client.Timeout = TimeSpan.FromMinutes(30);

            try
            {
                var days = Math.Min(_cfg.TrainDays, 90);

                // 1) Último treino (via /metrics)
                var metrics = await client.GetFromJsonAsync<MetricsResponseLite>($"{_cfg.BaseUrl}/metrics");
                DateTime? lastFinished = null;
                if (metrics != null && metrics.status == "ok" && !string.IsNullOrWhiteSpace(metrics.finished_at))
                {
                    if (DateTime.TryParse(metrics.finished_at, out var dt))
                        lastFinished = dt;
                }

                // Se nunca treinou, treina agora.
                if (lastFinished is null)
                {
                    await client.PostAsync($"{_cfg.BaseUrl}/train?days={days}", null);
                    await client.PostAsync($"{_cfg.BaseUrl}/series/rebuild?days={days}", null);
                    await client.PostAsync($"{_cfg.BaseUrl}/futures/update", null);
                    return;
                }

                var hoursSince = (DateTime.UtcNow - DateTime.SpecifyKind(lastFinished.Value, DateTimeKind.Utc)).TotalHours;
                if (hoursSince < _cfg.TrainMinHours)
                    return;

                // 2) MAPE rolling em futures (puxa só os últimos N pontos)
                var fut = await client.GetFromJsonAsync<FuturesResponseLite>($"{_cfg.BaseUrl}/futures?limit={_cfg.FuturesRollingN}");
                var mape = ComputeMapeRolling(fut);
                if (mape is null)
                    return;

                if (mape.Value >= _cfg.FuturesMapeThreshold)
                {
                    await client.PostAsync($"{_cfg.BaseUrl}/train?days={days}", null);
                    await client.PostAsync($"{_cfg.BaseUrl}/series/rebuild?days={days}", null);
                    await client.PostAsync($"{_cfg.BaseUrl}/futures/update", null);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "TrainDriftJob falhou");
            }
        }

        private double? ComputeMapeRolling(FuturesResponseLite? resp)
        {
            if (resp?.points == null || resp.points.Count == 0)
                return null;
            double sum = 0.0;
            int n = 0;
            foreach (var p in resp.points)
            {
                if (p.real_close is null || p.err_close is null)
                    continue;
                var real = Math.Abs(p.real_close.Value);
                if (real <= 0)
                    continue;
                sum += Math.Abs(p.err_close.Value) / real * 100.0;
                n++;
            }
            return n > 0 ? (sum / n) : null;
        }

        // DTOs mínimos para evitar acoplamento com models do backend
        private class MetricsResponseLite
        {
            public string? status { get; set; }
            public string? finished_at { get; set; }
        }

        private class FuturesResponseLite
        {
            public List<FutPointLite> points { get; set; } = new();
        }

        private class FutPointLite
        {
            public string? time { get; set; }
            public double? real_close { get; set; }
            public double? err_close { get; set; }
        }
    }
}



