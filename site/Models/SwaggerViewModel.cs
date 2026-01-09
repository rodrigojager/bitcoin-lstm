using System.Collections.Generic;

namespace TechChallenge.Models
{
    public class SwaggerViewModel
    {
        public IEnumerable<string> Docs { get; set; } = new List<string>();
        public string DocsUrl { get; set; } = "http://localhost:8000/docs";
    }
}


