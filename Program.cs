using AiDotNetPocApi.Controllers;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

builder.WebHost.ConfigureKestrel(options =>
    options.Limits.MaxRequestBodySize = long.MaxValue);

builder.Services.Configure<FormOptions>(options =>
    options.MultipartBodyLengthLimit = long.MaxValue);

builder.Services.AddControllers();
builder.Services.AddTransient<ISpamClassificationFacade, TransformerSpamClassificationFacade>();

var app = builder.Build();

app.MapControllers();

app.Run();
