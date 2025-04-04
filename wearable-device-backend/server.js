// server.js
const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors({
  origin: '*',  // Allow all origins for testing (restrict in production)
  credentials: true
}));
app.use(express.json());

const CLIENT_ID = process.env.FITBIT_CLIENT_ID;
const CLIENT_SECRET = process.env.FITBIT_CLIENT_SECRET;
const REDIRECT_URI = process.env.FITBIT_REDIRECT_URI;

// Debugging middleware to log all requests
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  console.log('Request Headers:', JSON.stringify(req.headers, null, 2));
  console.log('Request Body:', JSON.stringify(req.body, null, 2));
  next();
});

// Endpoint to exchange authorization code for tokens
app.post('/api/fitbit/token', async (req, res) => {
  console.log('=== FITBIT TOKEN EXCHANGE REQUEST ===');
  console.log('Environment Variables:');
  console.log('- CLIENT_ID:', CLIENT_ID);
  console.log('- CLIENT_SECRET:', CLIENT_SECRET ? 'Set (first 4 chars: ' + CLIENT_SECRET.substring(0, 4) + '...)' : 'NOT SET');
  console.log('- REDIRECT_URI:', REDIRECT_URI);
  
  try {
    const { code } = req.body;
    
    console.log('Received authorization code:', code ? (code.substring(0, 10) + '...') : 'MISSING');
    
    if (!code) {
      return res.status(400).json({ 
        success: false, 
        errors: ['Authorization code is required'] 
      });
    }

    // Create request parameters
    const params = new URLSearchParams({
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
      grant_type: 'authorization_code',
      redirect_uri: REDIRECT_URI,
      code: code
    }).toString();
    
    console.log('Token request parameters:', params.replace(CLIENT_SECRET, '[REDACTED]'));
    
    // Create Basic auth header
    const basicAuth = Buffer.from(`${CLIENT_ID}:${CLIENT_SECRET}`).toString('base64');
    console.log('Basic auth header created (not showing value)');
    
    console.log('Sending token request to Fitbit API...');
    
    // Exchange code for token with Fitbit API
    const tokenResponse = await axios.post(
      'https://api.fitbit.com/oauth2/token',
      params,
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Authorization': `Basic ${basicAuth}`
        },
      }
    );

    console.log('Fitbit API token response received:');
    console.log('- Status:', tokenResponse.status);
    console.log('- Headers:', JSON.stringify(tokenResponse.headers, null, 2));
    console.log('- Data:', JSON.stringify({
      ...tokenResponse.data,
      access_token: tokenResponse.data.access_token ? 'RECEIVED (redacted)' : 'MISSING',
      refresh_token: tokenResponse.data.refresh_token ? 'RECEIVED (redacted)' : 'MISSING'
    }, null, 2));

    // Return the tokens to the frontend
    return res.json({
      success: true,
      ...tokenResponse.data
    });
  } catch (error) {
    console.error('=== TOKEN EXCHANGE ERROR ===');
    console.error('Error occurred during token exchange:');
    console.error('- Message:', error.message);
    console.error('- Status:', error.response?.status);
    console.error('- Response data:', JSON.stringify(error.response?.data, null, 2));
    console.error('- Response headers:', JSON.stringify(error.response?.headers, null, 2));
    
    return res.status(error.response?.status || 500).json({
      success: false,
      errors: [error.response?.data?.errors || error.response?.data || 'Failed to exchange authorization code'],
      errorDetails: {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data
      }
    });
  }
});

// Proxy endpoint for Fitbit API calls
app.get('/api/fitbit/data', async (req, res) => {
    try {
      const { endpoint, access_token } = req.query;
      
      console.log('Requested endpoint:', endpoint);
      console.log('Access token:', access_token ? 'PRESENT (redacted)' : 'MISSING');
      
      if (!access_token) {
        return res.status(401).json({
          success: false,
          errors: ['Access token is required']
        });
      }
      
      if (!endpoint) {
        return res.status(400).json({
          success: false,
          errors: ['Endpoint is required']
        });
      }
      
      console.log('Sending data request to Fitbit API...');
      
      const apiResponse = await axios.get(
        `https://api.fitbit.com/1/user/-/${endpoint}`,
        {
          headers: {
            Authorization: `Bearer ${access_token}`,
          },
        }
      );
      
    
    console.log('Fitbit API data response received:');
    console.log('- Status:', apiResponse.status);
    console.log('- Headers:', JSON.stringify(apiResponse.headers, null, 2));
    console.log('- Data (sample):', Object.keys(apiResponse.data).join(', '));

    return res.json({
      success: true,
      data: apiResponse.data
    });
  } catch (error) {
    console.error('=== DATA REQUEST ERROR ===');
    console.error('Error occurred during data request:');
    console.error('- Message:', error.message);
    console.error('- Status:', error.response?.status);
    console.error('- Response data:', JSON.stringify(error.response?.data, null, 2));
    
    return res.status(error.response?.status || 500).json({
      success: false,
      errors: [error.response?.data?.errors || error.response?.data || 'Failed to fetch data from Fitbit'],
      errorDetails: {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data
      }
    });
  }
});

// Simple test endpoint
app.get('/api/test', (req, res) => {
  console.log('Test endpoint accessed');
  res.json({ 
    success: true, 
    message: 'Backend server is running correctly',
    environment: {
      clientIdSet: !!CLIENT_ID,
      clientSecretSet: !!CLIENT_SECRET,
      redirectUriSet: !!REDIRECT_URI
    }
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`=== SERVER STARTED ===`);
  console.log(`Server running on port ${PORT}`);
  console.log('Environment configuration:');
  console.log('- NODE_ENV:', process.env.NODE_ENV);
  console.log('- FITBIT_CLIENT_ID:', CLIENT_ID || 'NOT SET');
  console.log('- FITBIT_CLIENT_SECRET:', CLIENT_SECRET ? 'SET (hidden)' : 'NOT SET');
  console.log('- FITBIT_REDIRECT_URI:', REDIRECT_URI || 'NOT SET');
  console.log(`- Available endpoints:`);
  console.log(`  * POST http://localhost:${PORT}/api/fitbit/token`);
  console.log(`  * GET http://localhost:${PORT}/api/fitbit/data/:endpoint`);
  console.log(`  * GET http://localhost:${PORT}/api/test`);
});