using AiDotNetPocApi.Controllers;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

builder.WebHost.ConfigureKestrel(options =>
    options.Limits.MaxRequestBodySize = long.MaxValue);

builder.Services.Configure<FormOptions>(options =>
    options.MultipartBodyLengthLimit = long.MaxValue);

builder.Services.AddControllers();
// Singleton: the facade trains its FTTransformerNetwork once (lazily, on first
// request) and serves all subsequent requests from the cached trained model.
builder.Services.AddSingleton<ISpamClassificationFacade, TransformerSpamClassificationFacade>();

var app = builder.Build();

app.MapControllers();

app.Run();
