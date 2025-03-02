import Link from "next/link"
import { ArrowRight } from "lucide-react"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-8 md:p-24">
      <div className="max-w-4xl w-full text-center">
        <h1 className="text-4xl md:text-6xl font-bold mb-6">Black-Scholes Option Pricing Calculator</h1>
        <p className="text-lg md:text-xl mb-8 text-gray-600 dark:text-gray-300">
          Calculate Call and Put option prices using the Black-Scholes/Merton model with real-time stock data.
        </p>

        <Link
          href="/calculator"
          className="inline-flex items-center justify-center px-6 py-3 text-lg font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
        >
          Get Started <ArrowRight className="ml-2 h-5 w-5" />
        </Link>

        <div className="mt-16 p-6 bg-gray-100 dark:bg-gray-800 rounded-lg">
          <h2 className="text-2xl font-bold mb-4">About the Black-Scholes Model</h2>
          <p className="text-left mb-4">
            The Black-Scholes model is a mathematical model used to determine the theoretical price of European-style
            options. The model assumes that the price of heavily traded assets follows a geometric Brownian motion with
            constant drift and volatility.
          </p>

          <div className="text-left mb-4">
            <h3 className="text-xl font-semibold mb-2">The Black-Scholes Formula:</h3>
            <div className="p-4 bg-white dark:bg-gray-700 rounded overflow-auto">
              <p className="mb-2">For a call option:</p>
              <p className="font-mono">C = S₀N(d₁) - Ke⁻ʳᵗN(d₂)</p>

              <p className="mt-4 mb-2">For a put option:</p>
              <p className="font-mono">P = Ke⁻ʳᵗN(-d₂) - S₀N(-d₁)</p>

              <p className="mt-4 mb-2">Where:</p>
              <p className="font-mono">d₁ = [ln(S₀/K) + (r + σ²/2)t] / (σ√t)</p>
              <p className="font-mono">d₂ = d₁ - σ√t</p>
            </div>
          </div>

          <div className="text-left">
            <h3 className="text-xl font-semibold mb-2">Variables:</h3>
            <ul className="list-disc pl-5 space-y-1">
              <li>S₀ = Current stock price</li>
              <li>K = Strike price</li>
              <li>r = Risk-free interest rate</li>
              <li>t = Time to expiration (in years)</li>
              <li>σ = Volatility of the stock</li>
              <li>N() = Cumulative distribution function of the standard normal distribution</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  )
}

