import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  // This is the place to change the backend URL for API requests (currently configured for a docker network)
  rewrites: async () => {
    return [
      {
        source: '/api/:path*',
        destination: `http://backend:8000/api/:path*`,
        // TEST
        // destination: `http://localhost:8000/api/:path*`,
      },
    ]
  },
};

export default nextConfig;
