using System.Collections.Generic;

namespace TechChallenge.Models
{
    public class DocViewModel
    {
        public IEnumerable<string> Docs { get; set; }
        public string Selected { get; set; }
        public string Html { get; set; }
    }
}
