# PowerShell script for Vercel deployment

Write-Host "Starting Vercel deployment..." -ForegroundColor Green

# Login to Vercel
Write-Host "Logging in to Vercel..." -ForegroundColor Yellow
npx vercel login

# Add environment variable
Write-Host "Setting up environment variables..." -ForegroundColor Yellow
npx vercel env add AIPIPE_API_KEY production

# Deploy to production
Write-Host "Deploying to Vercel..." -ForegroundColor Yellow
npx vercel --prod

Write-Host "Deployment complete!" -ForegroundColor Green