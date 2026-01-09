using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using System.Net.Http;
using TechChallenge.Models;
using TechChallenge.Services;
using System.Text;
using System.Text.Json;

namespace TechChallenge.Controllers
{
    [Route("Swagger")]
    public class SwaggerController : Controller
    {
        private readonly MarkdownService _markdownService;
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly ApiSettings _apiSettings;
        private readonly ILogger<SwaggerController> _logger;

        public SwaggerController(
            MarkdownService markdownService,
            IHttpClientFactory httpClientFactory,
            IOptions<ApiSettings> apiOptions,
            ILogger<SwaggerController> logger)
        {
            _markdownService = markdownService;
            _httpClientFactory = httpClientFactory;
            _apiSettings = apiOptions.Value;
            _logger = logger;
        }

        [HttpGet("")]
        public IActionResult Index()
        {
            var docs = _markdownService.ListDocs().ToList();
            var vm = new SwaggerViewModel { Docs = docs };
            var baseUrl = string.IsNullOrWhiteSpace(_apiSettings.BaseUrl) ? "http://localhost:8000" : _apiSettings.BaseUrl.TrimEnd('/');
            try
            {
                var uri = new Uri(baseUrl);

                if (string.Equals(uri.Host, "pyapi", StringComparison.OrdinalIgnoreCase))
                {
                    var browserHost = Request.Host.Host;
                    vm.DocsUrl = $"{Request.Scheme}://{browserHost}:8000/docs";
                }
                else
                {
                    vm.DocsUrl = baseUrl + "/docs";
                }
            }
            catch
            {
                vm.DocsUrl = "http://localhost:8000/docs";
            }
            return View(vm);
        }

        [HttpGet("OpenApi")]
        [ResponseCache(NoStore = true, Location = ResponseCacheLocation.None)]
        public async Task<IActionResult> OpenApi()
        {
            try
            {
                var baseUrl = string.IsNullOrWhiteSpace(_apiSettings.BaseUrl) ? "http://pyapi:8000" : _apiSettings.BaseUrl;
                var url = baseUrl.TrimEnd('/') + "/openapi.json";
                var client = _httpClientFactory.CreateClient();
                var json = await client.GetStringAsync(url);

                // Reescrever servers para apontar ao proxy local do site (evita CORS e origem incorreta)
                var proxyBase = $"{Request.Scheme}://{Request.Host}/Swagger/Proxy";
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement.Clone();
                using var stream = new MemoryStream();
                using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false }))
                {
                    writer.WriteStartObject();
                    foreach (var prop in root.EnumerateObject())
                    {
                        if (prop.NameEquals("servers"))
                        {
                            writer.WritePropertyName("servers");
                            writer.WriteStartArray();
                            writer.WriteStartObject();
                            writer.WriteString("url", proxyBase);
                            writer.WriteEndObject();
                            writer.WriteEndArray();
                        }
                        else
                        {
                            prop.WriteTo(writer);
                        }
                    }
                    // Se não existia "servers", adiciona
                    if (!root.TryGetProperty("servers", out _))
                    {
                        writer.WritePropertyName("servers");
                        writer.WriteStartArray();
                        writer.WriteStartObject();
                        writer.WriteString("url", proxyBase);
                        writer.WriteEndObject();
                        writer.WriteEndArray();
                    }
                    writer.WriteEndObject();
                }
                var rewritten = stream.ToArray();
                return File(rewritten, "application/json; charset=utf-8");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Erro ao obter OpenAPI do pyapi");
                return StatusCode(502, new { message = "Erro ao obter OpenAPI do pyapi" });
            }
        }

        // Proxy para requests "Try it out" do Swagger UI (evita CORS)
        [Route("Proxy/{**path}")]
        [HttpGet, HttpPost, HttpPut, HttpDelete, HttpPatch, HttpHead, HttpOptions]
        public async Task<IActionResult> Proxy(string path)
        {
            var baseUrl = string.IsNullOrWhiteSpace(_apiSettings.BaseUrl) ? "http://localhost:8000" : _apiSettings.BaseUrl;
            var target = baseUrl.TrimEnd('/') + "/" + (path ?? string.Empty);
            var qs = Request.QueryString.HasValue ? Request.QueryString.Value : string.Empty;
            var targetUrl = target + qs;
            try
            {
                var method = new HttpMethod(Request.Method);
                using var reqMessage = new HttpRequestMessage(method, targetUrl);

                // Copia corpo (quando houver)
                if (Request.ContentLength.HasValue && Request.ContentLength.Value > 0)
                {
                    using var reader = new StreamReader(Request.Body, Encoding.UTF8, leaveOpen: true);
                    Request.Body.Position = 0;
                    var bodyStr = await reader.ReadToEndAsync();
                    reqMessage.Content = new StringContent(bodyStr, Encoding.UTF8, Request.ContentType ?? "application/json");
                }

                // Copia alguns headers (exceto Host)
                foreach (var header in Request.Headers)
                {
                    if (string.Equals(header.Key, "Host", StringComparison.OrdinalIgnoreCase)) continue;
                    if (!reqMessage.Headers.TryAddWithoutValidation(header.Key, header.Value.ToArray()))
                    {
                        reqMessage.Content?.Headers.TryAddWithoutValidation(header.Key, header.Value.ToArray());
                    }
                }

                var client = _httpClientFactory.CreateClient();
                using var resp = await client.SendAsync(reqMessage, HttpCompletionOption.ResponseHeadersRead);
                var respBytes = await resp.Content.ReadAsByteArrayAsync();
                var contentType = resp.Content.Headers.ContentType?.ToString() ?? "application/json";

                Response.StatusCode = (int)resp.StatusCode;
                foreach (var header in resp.Headers)
                {
                    Response.Headers[header.Key] = header.Value.ToArray();
                }
                foreach (var header in resp.Content.Headers)
                {
                    Response.Headers[header.Key] = header.Value.ToArray();
                }
                // Alguns headers não podem ser reescritos
                Response.Headers.Remove("transfer-encoding");
                return File(respBytes, contentType);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Erro no proxy Swagger para {Target}", targetUrl);
                return StatusCode(502, new { message = "Erro no proxy do Swagger" });
            }
        }
    }
}


