# Build stage
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy csproj and restore
COPY TechChallenge.csproj ./
RUN dotnet restore "TechChallenge.csproj"

# Copy the rest and build
COPY . .
RUN dotnet publish "TechChallenge.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS final
WORKDIR /app

# Expose default Kestrel port
EXPOSE 8080

# Copy build output
COPY --from=build /app/publish .

# Use ASPNETCORE_URLS to listen on 0.0.0.0:8080
ENV ASPNETCORE_URLS=http://+:8080

# Disable HTTPS redirection in container unless certs provided
ENV ASPNETCORE_ENVIRONMENT=Production

ENTRYPOINT ["dotnet", "TechChallenge.dll"]
