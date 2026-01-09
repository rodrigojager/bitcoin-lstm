using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using System.Net.Http.Json;
using TechChallenge.Models;

namespace TechChallenge.Controllers
{
public class ChartsController : Controller
{
private readonly IHttpClientFactory _http;
private readonly ApiSettings _cfg;

public ChartsController(IHttpClientFactory http, IOptions<ApiSettings> cfg)
{
_http = http;
_cfg = cfg.Value;
}

public IActionResult Index() => View();

[HttpGet]
public async Task<IActionResult> Series()
{
var client = _http.CreateClient();

var json = await client.GetStringAsync($"{_cfg.BaseUrl}/series/cached?fallback_days=90");
return Content(json, "application/json");
}

[HttpGet]
public async Task<IActionResult> Metrics()
{
var client = _http.CreateClient();
var json = await client.GetStringAsync($"{_cfg.BaseUrl}/metrics");
return Content(json, "application/json");
}

[HttpPost]
public async Task<IActionResult> FuturesUpdate()
{
var client = _http.CreateClient();
var resp = await client.PostAsync($"{_cfg.BaseUrl}/futures/update", null);
var json = await resp.Content.ReadAsStringAsync();
return Content(json, "application/json");
}

[HttpGet]
public async Task<IActionResult> Futures()
{
var client = _http.CreateClient();
var json = await client.GetStringAsync($"{_cfg.BaseUrl}/futures");
return Content(json, "application/json");
}
}
}
