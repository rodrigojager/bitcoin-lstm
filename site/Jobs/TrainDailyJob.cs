using Quartz;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Logging;
using TechChallenge.Models;

namespace TechChallenge.Jobs
{
    /// <summary>
    /// Treino diário fixo (Opção C): sempre executa /train e depois /series/rebuild.
    /// </summary>
    public class TrainDailyJob : IJob
    {
        private readonly IHttpClientFactory _http;
        private readonly ApiSettings _cfg;
        private readonly ILogger<TrainDailyJob> _logger;

        public TrainDailyJob(IHttpClientFactory http, IOptions<ApiSettings> cfg, ILogger<TrainDailyJob> logger)
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
                await client.PostAsync($"{_cfg.BaseUrl}/train?days={days}", null);
                await client.PostAsync($"{_cfg.BaseUrl}/series/rebuild?days={days}", null);
                await client.PostAsync($"{_cfg.BaseUrl}/futures/update", null);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "TrainDailyJob falhou ao chamar /train ou /series/rebuild");
            }
        }
    }
}



