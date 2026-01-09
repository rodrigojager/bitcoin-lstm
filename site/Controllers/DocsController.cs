using Microsoft.AspNetCore.Mvc;
using TechChallenge.Models;
using TechChallenge.Services;
using Microsoft.Extensions.Logging;

namespace TechChallenge.Controllers
{
    /// <summary>Lista arquivos Markdown da pasta Docs e exibe o conteúdo.</summary>
    public class DocsController : Controller
    {
        private readonly MarkdownService _service;
        private readonly ILogger<DocsController> _logger;

        // Serviços injetados pelo DI do ASP.NET Core
        public DocsController(MarkdownService markdownService, ILogger<DocsController> logger)
        {
            _logger = logger;
            _service = markdownService;
            _logger.LogInformation("DocsController inicializado com MarkdownService injetado");
        }

        // /Docs ou /Docs/Index?id=NomeDoArquivo
        public IActionResult Index(string? id = null)
        {
            try
            {
                var docs = _service.ListDocs().ToList();
                
                var first = id ?? docs.FirstOrDefault() ?? "";

                var vm = new DocViewModel
                {
                    Docs = docs,
                    Selected = first,
                    Html = _service.RenderHtml(first)
                };
                return View(vm);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Erro ao carregar documentação");
                return View(new DocViewModel
                {
                    Docs = new List<string>(),
                    Selected = "",
                    Html = $"<p>Erro ao carregar documentação: {ex.Message}</p>"
                });
            }
        }

        // /Docs/Content?id=NomeDoArquivo  (usado via AJAX)
        [HttpGet]
        public new IActionResult Content(string id)
        {
            try
            {
                _logger.LogInformation("Carregando conteúdo do documento: {Id}", id);
                var html = _service.RenderHtml(id);
                return PartialView("_MarkdownContent", html);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Erro ao carregar conteúdo do documento: {Id}", id);
                return PartialView("_MarkdownContent", $"<p>Erro ao carregar documento: {ex.Message}</p>");
            }
        }
    }
}
