using Markdig;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using System.Text;

namespace TechChallenge.Services
{
    /// <summary>Converte Markdown para HTML e lista arquivos .md na pasta Docs.</summary>
    public class MarkdownService
    {
        private readonly string _docsPath;
        private readonly MarkdownPipeline _pipeline;
        private readonly ILogger<MarkdownService> _logger;

        public MarkdownService(string docsPath, ILogger<MarkdownService> logger = null)
        {
            _docsPath = docsPath;
            _logger = logger;
            _pipeline = new MarkdownPipelineBuilder().UseAdvancedExtensions().Build();

            
            if (Directory.Exists(_docsPath))
            {
                var files = Directory.GetFiles(_docsPath, "*.md");
            }
        }

        // Retorna lista de nomes de arquivo (sem extensão) ordenados alfabeticamente.
        public IEnumerable<string> ListDocs()
        {
            if (!Directory.Exists(_docsPath))
            {
                _logger?.LogWarning("Pasta Docs não encontrada: {DocsPath}", _docsPath);
                return Enumerable.Empty<string>();
            }

            try
            {
                var files = Directory.GetFiles(_docsPath, "*.md")
                                    .Select(filePath => 
                                    {
                                        var fileName = Path.GetFileNameWithoutExtension(filePath);
                                        // Corrige encoding usando conversão de bytes
                                        var correctedName = FixEncoding(fileName);
                                        return correctedName;
                                    })
                                    .OrderBy(n => n)
                                    .ToList();
                
                return files;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Erro ao listar documentos");
                return Enumerable.Empty<string>();
            }
        }

        private string FixEncoding(string fileName)
        {
            try
            {
                // Converte a string para bytes usando o encoding atual
                var bytes = Encoding.Default.GetBytes(fileName);
                
                // Tenta diferentes encodings para corrigir
                var encodings = new[] 
                {
                    Encoding.UTF8,
                    Encoding.GetEncoding(1252), // Windows-1252
                    Encoding.GetEncoding(28591) // ISO-8859-1
                };
                
                foreach (var encoding in encodings)
                {
                    var corrected = encoding.GetString(bytes);
                    // Verifica se a string corrigida não tem caracteres de encoding incorreto
                    if (!corrected.Contains("├") && !corrected.Contains("з") && !corrected.Contains("г"))
                    {
                        return corrected;
                    }
                }
                
                return fileName;
            }
            catch
            {
                return fileName;
            }
        }

        // Lê o arquivo de nome dado (sem extensão) e devolve HTML pronto.
        public string RenderHtml(string name)
        {
            if (string.IsNullOrEmpty(name))
            {
                _logger?.LogWarning("Nome do documento é nulo ou vazio");
                return "<p>Nome do documento não especificado.</p>";
            }

            var file = Path.Combine(_docsPath, $"{name}.md");
            
            if (!System.IO.File.Exists(file))
            {
                _logger?.LogWarning("Arquivo não encontrado: {File}", file);
                return $"<p>Documento '{name}' não encontrado.</p><p>Caminho tentado: {file}</p>";
            }

            try
            {
                var md = System.IO.File.ReadAllText(file, Encoding.UTF8);
                return Markdown.ToHtml(md, _pipeline);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Erro ao ler arquivo: {File}", file);
                return $"<p>Erro ao ler documento '{name}': {ex.Message}</p>";
            }
        }
    }
}
