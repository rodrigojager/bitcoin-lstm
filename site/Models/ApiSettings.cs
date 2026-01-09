namespace TechChallenge.Models
{
    public class ApiSettings
    {
        public string BaseUrl { get; set; } = "http://pyapi:8000";
        public string IngestCron { get; set; } = "0 */5 * * * ?";

        // Opção C: treino diário fixo + checagem de drift
        public string TrainDailyCron { get; set; } = "0 0 0 * * ?";      // meia-noite UTC/local do container
        public string TrainDriftCron { get; set; } = "0 0 */1 * * ?";    // a cada 1h
        public int TrainDays { get; set; } = 90;
        public double TrainMinHours { get; set; } = 12.0;
        public int FuturesRollingN { get; set; } = 288; // ~24h em 5m
        public double FuturesMapeThreshold { get; set; } = 0.8;
        public bool RunBackfillOnStartup { get; set; } = true;
        public int BackfillDays { get; set; } = 90;
        public double ExpectedCoverageRatio { get; set; } = 0.80;
        public int BackfillSleepMs { get; set; } = 500;
        public int BackfillLimit { get; set; } = 1000;
    }
}
