using FluentValidation.AspNetCore;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using SmartWebAppAPI.Entity.Models;

using SmartWebAppAPI.Repositories;
using SmartWebAppAPI.Services;
using SmartWebAppAPI.Validators.AuthValidators;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.



builder.Services.AddDbContext<RepositoryContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddControllers()
        .AddFluentValidation(fv => fv.RegisterValidatorsFromAssemblyContaining<AuthValidationRegister>());
builder.Services.AddControllers();

// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddScoped<IRepositoryManager, RepositoryManager>();
builder.Services.AddScoped<IAuthRepository, AuthRepository>();
builder.Services.AddScoped<IServiceManager, ServiceManager>();
builder.Services.AddScoped<IAuthService,AuthManager>();


builder.Services.AddAutoMapper(typeof(Program));

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();
app.UseAuthentication();
app.MapControllers();




app.Run();
