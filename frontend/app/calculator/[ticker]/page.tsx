"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Loader2 } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

interface StockData {
  ticker: string
  current_price: number
  volatility: number
  company_name: string
}

interface OptionResult {
  ticker: string
  stock_price: number
  strike_price: number
  time_to_expiry: number
  risk_free_rate: number
  volatility: number
  call_price: number
  put_price: number
  d1: number
  d2: number
}

export default function TickerPage({ params }: { params: { ticker: string } }) {
  const { ticker } = params

  const [stockData, setStockData] = useState<StockData | null>(null)
  const [loading, setLoading] = useState(true)
  const [calculating, setCalculating] = useState(false)
  const [error, setError] = useState("")
  const [result, setResult] = useState<OptionResult | null>(null)

  // Form state
  const [strikePrice, setStrikePrice] = useState("")
  const [timeToExpiry, setTimeToExpiry] = useState("0.25") // 3 months default
  const [riskFreeRate, setRiskFreeRate] = useState("0.05") // 5% default
  const [volatility, setVolatility] = useState("")

  // Chart data
  const [chartData, setChartData] = useState<any[]>([])

  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/get_stock_price/${ticker}`,
        )

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || "Failed to fetch stock data")
        }

        const data = await response.json()
        setStockData(data)
        setStrikePrice(data.current_price.toString())
        setVolatility(data.volatility.toString())
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred")
      } finally {
        setLoading(false)
      }
    }

    fetchStockData()
  }, [ticker])

  const calculateOption = async () => {
    setCalculating(true)
    setError("")

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/calculate_option`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ticker,
          strike_price: Number.parseFloat(strikePrice),
          time_to_expiry: Number.parseFloat(timeToExpiry),
          risk_free_rate: Number.parseFloat(riskFreeRate),
          volatility: Number.parseFloat(volatility),
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Failed to calculate option price")
      }

      const data = await response.json()
      setResult(data)

      // Generate chart data for different expiry times
      const chartPoints = []
      const maxExpiry = Math.max(1, Number.parseFloat(timeToExpiry) * 2) // Up to 2x the selected expiry or at least 1 year

      for (let t = 0.05; t <= maxExpiry; t += maxExpiry / 20) {
        const d1 =
          (Math.log(data.stock_price / data.strike_price) + (data.risk_free_rate + 0.5 * data.volatility ** 2) * t) /
          (data.volatility * Math.sqrt(t))

        const d2 = d1 - data.volatility * Math.sqrt(t)

        // Calculate call and put prices
        const callPrice =
          data.stock_price * normalCDF(d1) - data.strike_price * Math.exp(-data.risk_free_rate * t) * normalCDF(d2)

        const putPrice =
          data.strike_price * Math.exp(-data.risk_free_rate * t) * normalCDF(-d2) - data.stock_price * normalCDF(-d1)

        chartPoints.push({
          expiry: t.toFixed(2),
          call: callPrice.toFixed(2),
          put: putPrice.toFixed(2),
        })
      }

      setChartData(chartPoints)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setCalculating(false)
    }
  }

  // Standard normal cumulative distribution function
  const normalCDF = (x: number) => {
    const t = 1 / (1 + 0.2316419 * Math.abs(x))
    const d = 0.3989423 * Math.exp((-x * x) / 2)
    let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    if (x > 0) {
      prob = 1 - prob
    }
    return prob
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  if (error && !stockData) {
    return (
      <div className="container mx-auto py-10 px-4 text-center">
        <h1 className="text-3xl font-bold mb-4">Error</h1>
        <p className="text-red-500">{error}</p>
        <Button className="mt-4" onClick={() => window.history.back()}>
          Go Back
        </Button>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold mb-2 text-center">
        {stockData?.company_name} ({ticker})
      </h1>
      <p className="text-center mb-8">Current Price: ${stockData?.current_price.toFixed(2)}</p>

      <div className="grid md:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>Option Parameters</CardTitle>
            <CardDescription>Enter the parameters to calculate option prices</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="stockPrice">Stock Price ($)</Label>
                <Input id="stockPrice" value={stockData?.current_price.toFixed(2) || ""} disabled />
              </div>

              <div className="space-y-2">
                <Label htmlFor="strikePrice">Strike Price ($)</Label>
                <Input
                  id="strikePrice"
                  value={strikePrice}
                  onChange={(e) => setStrikePrice(e.target.value)}
                  type="number"
                  step="0.01"
                  min="0.01"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="timeToExpiry">Time to Expiry (years)</Label>
                <Input
                  id="timeToExpiry"
                  value={timeToExpiry}
                  onChange={(e) => setTimeToExpiry(e.target.value)}
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="10"
                />
                <p className="text-xs text-gray-500">Examples: 0.25 (3 months), 0.5 (6 months), 1 (1 year)</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="riskFreeRate">Risk-Free Rate</Label>
                <Input
                  id="riskFreeRate"
                  value={riskFreeRate}
                  onChange={(e) => setRiskFreeRate(e.target.value)}
                  type="number"
                  step="0.001"
                  min="0"
                  max="0.2"
                />
                <p className="text-xs text-gray-500">Example: 0.05 for 5%</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="volatility">Volatility</Label>
                <Input
                  id="volatility"
                  value={volatility}
                  onChange={(e) => setVolatility(e.target.value)}
                  type="number"
                  step="0.001"
                  min="0.001"
                />
                <p className="text-xs text-gray-500">Historical volatility: {stockData?.volatility.toFixed(4)}</p>
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button onClick={calculateOption} disabled={calculating} className="w-full">
              {calculating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Calculating...
                </>
              ) : (
                "Calculate Option Prices"
              )}
            </Button>
          </CardFooter>
        </Card>

        <div className="space-y-8">
          {result && (
            <Card>
              <CardHeader>
                <CardTitle>Option Prices</CardTitle>
                <CardDescription>Black-Scholes model results</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <h3 className="font-semibold mb-2">Call Option</h3>
                    <p className="text-2xl font-bold">${result.call_price.toFixed(2)}</p>
                  </div>

                  <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                    <h3 className="font-semibold mb-2">Put Option</h3>
                    <p className="text-2xl font-bold">${result.put_price.toFixed(2)}</p>
                  </div>
                </div>

                <div className="mt-4 space-y-2 text-sm">
                  <p>
                    <span className="font-semibold">d₁:</span> {result.d1.toFixed(4)}
                  </p>
                  <p>
                    <span className="font-semibold">d₂:</span> {result.d2.toFixed(4)}
                  </p>
                  <p>
                    <span className="font-semibold">N(d₁):</span> {normalCDF(result.d1).toFixed(4)}
                  </p>
                  <p>
                    <span className="font-semibold">N(d₂):</span> {normalCDF(result.d2).toFixed(4)}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {chartData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Option Price vs Time to Expiry</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{
                        top: 5,
                        right: 30,
                        left: 20,
                        bottom: 5,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="expiry"
                        label={{ value: "Time to Expiry (years)", position: "insideBottomRight", offset: -10 }}
                      />
                      <YAxis label={{ value: "Option Price ($)", angle: -90, position: "insideLeft" }} />
                      <Tooltip formatter={(value) => [`$${value}`, ""]} />
                      <Legend />
                      <Line type="monotone" dataKey="call" name="Call Option" stroke="#22c55e" activeDot={{ r: 8 }} />
                      <Line type="monotone" dataKey="put" name="Put Option" stroke="#ef4444" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

