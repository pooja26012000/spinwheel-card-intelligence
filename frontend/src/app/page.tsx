'use client'

import { useState } from 'react'

const API_URL = 'https://spinwheel-api-119676722067.us-central1.run.app'

function formatAnswer(text: string) {
  return text
    .replace(/\(Chunk \d+(?:,\s*Chunk \d+)*\)/g, '')     // remove (Chunk 1), (Chunk 2) etc
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')      // **bold**
    .replace(/\*(.+?)\*/g, '<em>$1</em>')                  // *italic*
    .replace(/^\* (.+)$/gm, '<li>$1</li>')                 // * bullet points
    .replace(/(<li>[\s\S]*<\/li>)/, '<ul class="list-disc list-inside space-y-1 my-2">$1</ul>')
    .replace(/\n/g, '<br />')                              // line breaks
}

export default function Home() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleAsk() {
    if (!question.trim()) return

    setLoading(true)
    setAnswer('')
    setSources([])
    setError('')

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()

      // Extract sources from the answer text
      const sourcesMatch = data.answer.match(/Sources:\s*\[([^\]]+)\]/)
      const extractedSources = sourcesMatch
        ? sourcesMatch[1].split(',').map((s: string) => s.trim())
        : []

      // Remove the "Sources: [...]" line from the displayed answer
      const cleanAnswer = data.answer.replace(/\s*Sources:\s*\[([^\]]+)\]/, '').trim()

      setAnswer(cleanAnswer)
      setSources(extractedSources)
    } catch (err) {
      setError('Something went wrong. Is the API running?')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen max-w-2xl mx-auto px-4 py-16">

      {/* Header */}
      <div className="mb-10">
        <h1 className="text-2xl font-semibold text-white mb-1">
          SpinWheel Card Intelligence
        </h1>
        <p className="text-sm text-gray-400">
          Ask anything about sports card grading and valuation
        </p>
      </div>

      {/* Input */}
      <div className="mb-6">
        <textarea
          className="w-full bg-gray-900 border border-gray-700 rounded-lg p-4 text-white text-sm resize-none focus:border-yellow-600 focus:outline-none"
          rows={3}
          placeholder="e.g. What is a PSA 10 1986 Fleer Michael Jordan worth?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button
          className="mt-3 bg-yellow-600 hover:bg-yellow-500 text-black font-medium px-6 py-2 rounded-lg text-sm transition-colors disabled:opacity-50"
          onClick={handleAsk}
          disabled={loading}
        >
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-4">
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* Answer */}
      {answer && (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <p className="text-sm text-gray-400 mb-3">Answer</p>
          <p
            className="text-white text-sm leading-relaxed"
            dangerouslySetInnerHTML={{ __html: formatAnswer(answer) }}
          />

          {/* Sources */}
          {sources.length > 0 && (
            <div className="mt-6 pt-4 border-t border-gray-700">
              <p className="text-xs text-gray-500 mb-2">Sources</p>
              <div className="flex flex-wrap gap-2">
                {sources.map((src, i) => (
                  <span
                    key={i}
                    className="text-xs bg-yellow-900 text-yellow-300 border border-yellow-700 px-2 py-1 rounded font-mono"
                  >
                    {src}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

    </main>
  )
}
