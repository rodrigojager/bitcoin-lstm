using Quartz;
using Microsoft.Extensions.Options;
using TechChallenge.Models;
using Microsoft.Extensions.Logging;

namespace TechChallenge.Jobs
{
    public class IngestJob : IJob
    {
        private readonly IHttpClientFactory _http;
        private readonly ApiSettings _cfg;
        private readonly ILogger<IngestJob> _logger;

        public IngestJob(IHttpClientFactory http, IOptions<ApiSettings> cfg, ILogger<IngestJob> logger)
        {
            _http = http;
            _cfg = cfg.Value;
            _logger = logger;
        }

        public async Task Execute(IJobExecutionContext context)
        {
            var client = _http.CreateClient();
            try
            {
                client.Timeout = TimeSpan.FromMinutes(5);
                await client.PostAsync($"{_cfg.BaseUrl}/ingest", null);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "IngestJob falhou ao chamar /ingest");
            }
        }
    }
}
