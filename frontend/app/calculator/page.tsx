"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Loader2, Search } from "lucide-react"

export default function Calculator() {
  const router = useRouter()
  const [ticker, setTicker] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!ticker) {
      setError("Please enter a stock ticker")
      return
    }

    setIsLoading(true)
    setError("")

    try {
      // Validate the ticker by fetching stock data
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/get_stock_price/${ticker}`,
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Failed to fetch stock data")
      }

      // If successful, navigate to the options page
      router.push(`/calculator/${ticker}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold mb-6 text-center">Option Calculator</h1>

      <Card className="max-w-md mx-auto">
        <CardHeader>
          <CardTitle>Enter Stock Ticker</CardTitle>
          <CardDescription>Enter a valid stock symbol (e.g., AAPL, MSFT, TSLA)</CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="ticker">Stock Ticker</Label>
                <div className="flex space-x-2">
                  <Input
                    id="ticker"
                    placeholder="e.g. AAPL"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    className="flex-1"
                  />
                  <Button type="submit" disabled={isLoading}>
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  </Button>
                </div>
                {error && <p className="text-sm text-red-500">{error}</p>}
              </div>
            </div>
          </CardContent>
        </form>
        <CardFooter className="flex justify-center text-sm text-gray-500">
          Stock data provided by Yahoo Finance
        </CardFooter>
      </Card>
    </div>
  )
}

