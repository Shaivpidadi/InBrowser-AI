"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { motion, AnimatePresence } from "framer-motion"
import { Diamond, Bomb, Coins, Brain, TrendingUp, Sparkles } from "lucide-react"
import * as tf from '@tensorflow/tfjs'

// Types
type GameState = "idle" | "playing" | "lost" | "won"
type AIMode = "behavior" | "strategy" | "pattern"
type TileState = "hidden" | "gem" | "bomb"

interface GameMove {
  tileIndex: number
  timestamp: number
  gameState: number[] // One-hot encoded revealed tiles
  wasSuccessful: boolean
  minesCount: number
  revealedCount: number
}

interface PatternState {
  round: number
  lastPattern: number[]
}

// Utility Functions
function calculatePayout(bet: number, mines: number, diamonds: number): number {
  let M = 1
  for (let i = 0; i < diamonds; i++) {
    M = (M * (25 - i)) / (25 - mines - i)
  }
  return bet * M
}

function extractGameStateVector(revealedTiles: number[], grid: TileState[]): number[] {
  // Create a 25-element vector: 0 = hidden, 1 = revealed gem, -1 = revealed bomb
  return grid.map((tile, idx) => {
    if (!revealedTiles.includes(idx)) return 0
    return tile === "gem" ? 1 : -1
  })
}

function getTileNeighbors(index: number): number[] {
  const row = Math.floor(index / 5)
  const col = index % 5
  const neighbors: number[] = []
  
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue
      const newRow = row + dr
      const newCol = col + dc
      if (newRow >= 0 && newRow < 5 && newCol >= 0 && newCol < 5) {
        neighbors.push(newRow * 5 + newCol)
      }
    }
  }
  return neighbors
}

// IndexedDB Storage
const DB_NAME = "MiningGameDB"
const MOVES_STORE = "playerMoves"
const STRATEGY_STORE = "strategyData"
const PATTERN_STORE = "patternData"

async function initDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1)
    
    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)
    
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(MOVES_STORE)) {
        db.createObjectStore(MOVES_STORE, { autoIncrement: true })
      }
      if (!db.objectStoreNames.contains(STRATEGY_STORE)) {
        db.createObjectStore(STRATEGY_STORE, { autoIncrement: true })
      }
      if (!db.objectStoreNames.contains(PATTERN_STORE)) {
        db.createObjectStore(PATTERN_STORE, { keyPath: "round" })
      }
    }
  })
}

async function saveData(store: string, data: any): Promise<void> {
  const db = await initDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([store], "readwrite")
    const objectStore = transaction.objectStore(store)
    const request = objectStore.add(data)
    request.onsuccess = () => resolve()
    request.onerror = () => reject(request.error)
  })
}

async function loadAllData(store: string): Promise<any[]> {
  const db = await initDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([store], "readonly")
    const objectStore = transaction.objectStore(store)
    const request = objectStore.getAll()
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function savePatternState(state: PatternState): Promise<void> {
  const db = await initDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([PATTERN_STORE], "readwrite")
    const objectStore = transaction.objectStore(PATTERN_STORE)
    const request = objectStore.put(state)
    request.onsuccess = () => resolve()
    request.onerror = () => reject(request.error)
  })
}

async function loadPatternState(): Promise<PatternState | null> {
  const db = await initDB()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([PATTERN_STORE], "readonly")
    const objectStore = transaction.objectStore(PATTERN_STORE)
    const request = objectStore.get(1)
    request.onsuccess = () => resolve(request.result || null)
    request.onerror = () => reject(request.error)
  })
}

export default function MiningGame() {
  // Game State
  const [amount, setAmount] = useState("1")
  const [mines, setMines] = useState("3")
  const [gameState, setGameState] = useState<GameState>("idle")
  const [revealedTiles, setRevealedTiles] = useState<number[]>([])
  const [coins, setCoins] = useState(100)
  const [currentBet, setCurrentBet] = useState(0)
  const [currentPayout, setCurrentPayout] = useState(0)
  const [grid, setGrid] = useState<TileState[]>(Array(25).fill("hidden"))
  
  // AI State
  const [aiMode, setAiMode] = useState<AIMode>("behavior")
  const [isModelReady, setIsModelReady] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState<string>("")
  const [totalMoves, setTotalMoves] = useState(0)
  
  // Model References
  const behaviorModelRef = useRef<tf.LayersModel | null>(null)
  const strategyModelRef = useRef<tf.LayersModel | null>(null)
  const patternModelRef = useRef<tf.LayersModel | null>(null)
  
  // Game History
  const gameMovesRef = useRef<GameMove[]>([])
  const currentGameStartRef = useRef<number>(Date.now())
  const patternStateRef = useRef<PatternState>({ round: 0, lastPattern: [] })

  // Initialize coins from localStorage
  useEffect(() => {
    const savedCoins = localStorage.getItem("miningGameCoins")
    if (savedCoins) {
      setCoins(Number.parseFloat(savedCoins))
    }
  }, [])

  useEffect(() => {
    localStorage.setItem("miningGameCoins", coins.toString())
  }, [coins])

  // Initialize AI Models
  useEffect(() => {
    async function initializeModels() {
      try {
        setTrainingProgress("Initializing AI models...")
        
        // Model 1: Player Behavior Prediction
        // Input: Last 10 clicks (tile indices), Output: Next likely tile
        try {
          const savedBehavior = await tf.loadLayersModel("indexeddb://behavior-model")
          behaviorModelRef.current = savedBehavior
          console.log("✅ Loaded behavior model")
        } catch {
          const behaviorModel = tf.sequential()
          behaviorModel.add(tf.layers.dense({ inputShape: [10], units: 64, activation: "relu" }))
          behaviorModel.add(tf.layers.dropout({ rate: 0.3 }))
          behaviorModel.add(tf.layers.dense({ units: 32, activation: "relu" }))
          behaviorModel.add(tf.layers.dense({ units: 25, activation: "softmax" }))
          behaviorModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: "sparseCategoricalCrossentropy",
            metrics: ["accuracy"]
          })
          behaviorModelRef.current = behaviorModel
          console.log("✅ Created new behavior model")
        }
        
        // Model 2: Optimal Strategy Learning
        // Input: Current game state (25 tiles + mines count), Output: Safety score per tile
        try {
          const savedStrategy = await tf.loadLayersModel("indexeddb://strategy-model")
          strategyModelRef.current = savedStrategy
          console.log("✅ Loaded strategy model")
        } catch {
          const strategyModel = tf.sequential()
          strategyModel.add(tf.layers.dense({ inputShape: [26], units: 128, activation: "relu" }))
          strategyModel.add(tf.layers.dropout({ rate: 0.3 }))
          strategyModel.add(tf.layers.dense({ units: 64, activation: "relu" }))
          strategyModel.add(tf.layers.dense({ units: 25, activation: "sigmoid" }))
          strategyModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
          })
          strategyModelRef.current = strategyModel
          console.log("✅ Created new strategy model")
        }
        
        // Model 3: Pattern Recognition
        // Input: Round number + last pattern, Output: Bomb positions
        try {
          const savedPattern = await tf.loadLayersModel("indexeddb://pattern-model")
          patternModelRef.current = savedPattern
          console.log("✅ Loaded pattern model")
        } catch {
          const patternModel = tf.sequential()
          patternModel.add(tf.layers.dense({ inputShape: [26], units: 64, activation: "relu" }))
          patternModel.add(tf.layers.dense({ units: 32, activation: "relu" }))
          patternModel.add(tf.layers.dense({ units: 25, activation: "sigmoid" }))
          patternModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
          })
          patternModelRef.current = patternModel
          console.log("✅ Created new pattern model")
        }
        
        // Load pattern state
        const savedPatternState = await loadPatternState()
        if (savedPatternState) {
          patternStateRef.current = savedPatternState
        }
        
        setIsModelReady(true)
        setTrainingProgress("")
      } catch (error) {
        console.error("Model initialization error:", error)
        setTrainingProgress("Error initializing models")
      }
    }

    initializeModels()
  }, [])

  // MODE 1: Train Behavior Model (Silent)
  async function trainBehaviorModel() {
    if (!behaviorModelRef.current) return
    
    const allMoves = await loadAllData(MOVES_STORE)
    if (allMoves.length < 10) {
      console.log("Need 10+ total moves to train")
      return
    }

    console.log("Training behavior model on", allMoves.length, "moves...")
    
    try {
      const sequences: number[][] = []
      const labels: number[] = []
      
      // Sliding window over ALL moves
      for (let i = 0; i < allMoves.length - 10; i++) {
        const sequence = allMoves.slice(i, i + 10).map((m: GameMove) => m.tileIndex)
        const nextMove = allMoves[i + 10].tileIndex
        sequences.push(sequence)
        labels.push(nextMove)
      }
      
      const X = tf.tensor2d(sequences)
      const y = tf.tensor1d(labels, "int32")
      
      await behaviorModelRef.current.fit(X, y, {
        epochs: 10,
        batchSize: 16,
        shuffle: true,
        validationSplit: 0.2,
        verbose: 0
      })
      
      await behaviorModelRef.current.save("indexeddb://behavior-model")
      console.log("✅ Behavior model trained")
      
      X.dispose()
      y.dispose()
    } catch (error) {
      console.error("Training error:", error)
    }
  }

  // MODE 2: Train Strategy Model (Silent)
  async function trainStrategyModel() {
    if (!strategyModelRef.current) return
    
    const allMoves = await loadAllData(STRATEGY_STORE)
    if (allMoves.length < 10) {
      console.log("Need 10+ total moves to train")
      return
    }

    console.log("Training strategy model on", allMoves.length, "moves...")
    
    try {
      const X_data: number[][] = []
      const y_data: number[][] = []
      
      for (const move of allMoves) {
        X_data.push([...move.gameState, move.minesCount / 25])
        
        const target = Array(25).fill(0)
        if (move.wasSuccessful) {
          target[move.tileIndex] = 1
        }
        y_data.push(target)
      }
      
      const X = tf.tensor2d(X_data)
      const y = tf.tensor2d(y_data)
      
      await strategyModelRef.current.fit(X, y, {
        epochs: 15,
        batchSize: 16,
        shuffle: true,
        validationSplit: 0.2,
        verbose: 0
      })
      
      await strategyModelRef.current.save("indexeddb://strategy-model")
      console.log("✅ Strategy model trained")
      
      X.dispose()
      y.dispose()
    } catch (error) {
      console.error("Training error:", error)
    }
  }

  // MODE 3: Train Pattern Model (Silent)
  async function trainPatternModel() {
    if (!patternModelRef.current) return
    
    const allPatterns = await loadAllData(PATTERN_STORE)
    if (allPatterns.length < 10) {
      console.log("Need 10+ total pattern games to train")
      return
    }

    console.log("Training pattern model on", allPatterns.length, "games...")
    
    try {
      const X_data: number[][] = []
      const y_data: number[][] = []
      
      for (let i = 0; i < allPatterns.length - 1; i++) {
        const current = allPatterns[i]
        const next = allPatterns[i + 1]
        
        const lastPattern = current.lastPattern.length === 25 ? current.lastPattern : Array(25).fill(0)
        X_data.push([current.round / 100, ...lastPattern])
        
        const target = Array(25).fill(0)
        next.lastPattern.forEach((pos: number) => {
          if (pos < 25) target[pos] = 1
        })
        y_data.push(target)
      }
      
      const X = tf.tensor2d(X_data)
      const y = tf.tensor2d(y_data)
      
      await patternModelRef.current.fit(X, y, {
        epochs: 20,
        batchSize: 8,
        shuffle: true,
        verbose: 0
      })
      
      await patternModelRef.current.save("indexeddb://pattern-model")
      console.log("✅ Pattern model trained")
      
      X.dispose()
      y.dispose()
    } catch (error) {
      console.error("Training error:", error)
    }
  }

  // Update total moves count
  useEffect(() => {
    async function updateMoveCount() {
      const moves = await loadAllData(MOVES_STORE)
      setTotalMoves(moves.length)
    }
    updateMoveCount()
  }, [gameState])

  // Pattern generation for Mode 3
  function generatePatternBombs(): number[] {
    const numMines = Number.parseInt(mines)
    const round = patternStateRef.current.round
    
    const patterns = [
      [0, 6, 12, 18, 24], // Diagonal
      [2, 10, 12, 14, 22], // Cross
      [0, 4, 20, 24, 12], // Corners
      [0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21, 20, 15, 10, 5], // Border
      [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24], // Checkerboard
      [6, 7, 8, 11, 12, 13, 16, 17, 18], // Center cluster
    ]
    
    const patternIndex = round % patterns.length
    const pattern = patterns[patternIndex]
    const bombPositions: number[] = []
    const shuffled = [...pattern].sort(() => Math.random() - 0.5)
    
    for (let i = 0; i < Math.min(numMines, shuffled.length); i++) {
      bombPositions.push(shuffled[i])
    }
    
    while (bombPositions.length < numMines) {
      const pos = Math.floor(Math.random() * 25)
      if (!bombPositions.includes(pos)) {
        bombPositions.push(pos)
      }
    }
    
    patternStateRef.current.lastPattern = bombPositions
    return bombPositions
  }

  // Game Logic
  const initializeGame = () => {
    const betAmount = Number.parseFloat(amount)
    if (isNaN(betAmount) || betAmount <= 0) {
      alert("Please enter a valid bet amount.")
      return
    }
    if (coins < betAmount) {
      alert("Not enough coins to place this bet!")
      return
    }

    setCoins(prevCoins => prevCoins - betAmount)
    setCurrentBet(betAmount)
    setCurrentPayout(betAmount)

    const newGrid: TileState[] = Array(25).fill("hidden")
    let bombPositions: number[] = []
    
    if (aiMode === "pattern") {
      bombPositions = generatePatternBombs()
      patternStateRef.current.round++
      savePatternState(patternStateRef.current)
    } else {
      const numMines = Number.parseInt(mines)
      while (bombPositions.length < numMines) {
        const position = Math.floor(Math.random() * 25)
        if (!bombPositions.includes(position)) {
          bombPositions.push(position)
        }
      }
    }

    bombPositions.forEach(pos => {
      newGrid[pos] = "bomb"
    })

    for (let i = 0; i < 25; i++) {
      if (newGrid[i] === "hidden") {
        newGrid[i] = "gem"
      }
    }

    setGrid(newGrid)
    setRevealedTiles([])
    setGameState("playing")
    currentGameStartRef.current = Date.now()
    gameMovesRef.current = []
  }

  const handleTileClick = async (index: number) => {
    if (gameState !== "playing" || revealedTiles.includes(index)) return

    const clickTime = Date.now()
    const newRevealedTiles = [...revealedTiles, index]
    setRevealedTiles(newRevealedTiles)

    const wasSuccessful = grid[index] === "gem"
    
    const move: GameMove = {
      tileIndex: index,
      timestamp: clickTime - currentGameStartRef.current,
      gameState: extractGameStateVector(revealedTiles, grid),
      wasSuccessful,
      minesCount: Number.parseInt(mines),
      revealedCount: revealedTiles.length
    }
    
    gameMovesRef.current.push(move)
    
    if (aiMode === "behavior") {
      await saveData(MOVES_STORE, move)
    } else if (aiMode === "strategy") {
      await saveData(STRATEGY_STORE, move)
    }

    if (grid[index] === "bomb") {
      setGameState("lost")
      setRevealedTiles([...Array(25).keys()])
      setCurrentPayout(0)
      
      setTimeout(() => {
        if (aiMode === "behavior") trainBehaviorModel()
        else if (aiMode === "strategy") trainStrategyModel()
        else if (aiMode === "pattern") trainPatternModel()
      }, 1000)
    } else {
      const safeReveals = newRevealedTiles.length
      const newPayout = calculatePayout(currentBet, Number.parseInt(mines), safeReveals)
      setCurrentPayout(newPayout)

      if (safeReveals === 25 - Number.parseInt(mines)) {
        setGameState("won")
        setRevealedTiles([...Array(25).keys()])
        setCoins(prevCoins => prevCoins + newPayout)
        
        setTimeout(() => {
          if (aiMode === "behavior") trainBehaviorModel()
          else if (aiMode === "strategy") trainStrategyModel()
          else if (aiMode === "pattern") trainPatternModel()
        }, 1000)
      }
    }
  }

  const handleCashout = () => {
    if (gameState === "playing") {
      setCoins(prevCoins => prevCoins + currentPayout)
      setGameState("won")
      setRevealedTiles([...Array(25).keys()])
    }
  }

  const handleAmountChange = (operation: "half" | "double") => {
    const currentAmount = Number.parseFloat(amount)
    const newAmount = operation === "half" ? currentAmount / 2 : currentAmount * 2
    setAmount(Math.max(0.1, newAmount).toFixed(2))
  }

  const claimCoins = () => {
    setCoins(prevCoins => prevCoins + 100)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">AI Mining Game</h1>
          <p className="text-slate-400">AI learns from your gameplay silently in the background</p>
        </div>

        {/* AI Mode Selector */}
        <div className="bg-slate-800 rounded-xl p-6 mb-6">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Brain className="w-6 h-6" />
            AI Mode
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={() => setAiMode("behavior")}
              className={`p-4 rounded-lg border-2 transition-all ${
                aiMode === "behavior"
                  ? "border-blue-500 bg-blue-500/20"
                  : "border-slate-700 hover:border-slate-600"
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5" />
                <span className="font-bold">Player Behavior</span>
              </div>
              <p className="text-sm text-slate-400">
                AI learns YOUR clicking patterns silently. Builds a model of your personal strategy.
              </p>
            </button>

            <button
              onClick={() => setAiMode("strategy")}
              className={`p-4 rounded-lg border-2 transition-all ${
                aiMode === "strategy"
                  ? "border-green-500 bg-green-500/20"
                  : "border-slate-700 hover:border-slate-600"
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-5 h-5" />
                <span className="font-bold">Optimal Strategy</span>
              </div>
              <p className="text-sm text-slate-400">
                AI analyzes what works and what doesn't. Learns the safest moves over time.
              </p>
            </button>

            <button
              onClick={() => setAiMode("pattern")}
              className={`p-4 rounded-lg border-2 transition-all ${
                aiMode === "pattern"
                  ? "border-purple-500 bg-purple-500/20"
                  : "border-slate-700 hover:border-slate-600"
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-5 h-5" />
                <span className="font-bold">Pattern Recognition</span>
              </div>
              <p className="text-sm text-slate-400">
                Bombs follow learnable patterns. AI recognizes sequences across multiple games.
              </p>
            </button>
          </div>
        </div>

        {/* Main Game Area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Game Grid */}
          <div className="bg-slate-800 rounded-xl p-6">
            <div className="grid grid-cols-5 gap-2">
              {grid.map((tile, index) => (
                <motion.button
                  key={index}
                  className={`aspect-square rounded-lg p-4 flex items-center justify-center
                    ${revealedTiles.includes(index) ? "bg-slate-700" : "bg-slate-700 hover:bg-slate-600"}`}
                  onClick={() => handleTileClick(index)}
                  whileHover={{ scale: gameState === "playing" ? 1.05 : 1 }}
                  whileTap={{ scale: gameState === "playing" ? 0.95 : 1 }}
                  disabled={gameState !== "playing"}
                >
                  <AnimatePresence>
                    {revealedTiles.includes(index) && (
                      <motion.div
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        exit={{ scale: 0 }}
                        transition={{ type: "spring", stiffness: 200, damping: 15 }}
                      >
                        {tile === "gem" ? (
                          <Diamond className="w-8 h-8 text-green-500" />
                        ) : (
                          <Bomb className="w-8 h-8 text-red-500" />
                        )}
                      </motion.div>
                    )}
                    {!revealedTiles.includes(index) && (
                      <Diamond className="w-8 h-8 text-slate-600" />
                    )}
                  </AnimatePresence>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Controls */}
          <div className="space-y-6">
            <div className="bg-slate-800 rounded-xl p-6 space-y-4">
              <div className="space-y-2">
                <label className="text-slate-400 text-sm">Bet Amount</label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Input
                      type="number"
                      value={amount}
                      onChange={(e) => setAmount(e.target.value)}
                      className="bg-slate-700 border-slate-600 pl-8"
                      step="0.1"
                      min="0.1"
                    />
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-green-500">$</span>
                  </div>
                  <Button
                    variant="outline"
                    className="border-slate-600 hover:bg-slate-700"
                    onClick={() => handleAmountChange("half")}
                  >
                    ½
                  </Button>
                  <Button
                    variant="outline"
                    className="border-slate-600 hover:bg-slate-700"
                    onClick={() => handleAmountChange("double")}
                  >
                    2×
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-slate-400 text-sm">Number of Mines</label>
                <Select value={mines} onValueChange={setMines}>
                  <SelectTrigger className="bg-slate-700 border-slate-600">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[1, 2, 3, 4, 5, 6, 7, 8].map((num) => (
                      <SelectItem key={num} value={num.toString()}>
                        {num} {num === 1 ? "Mine" : "Mines"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {gameState === "idle" && (
                <Button
                  className="w-full bg-green-500 hover:bg-green-600 text-white py-6 text-lg font-bold"
                  onClick={initializeGame}
                  disabled={!isModelReady}
                >
                  {isModelReady ? "Start Game" : "Loading AI..."}
                </Button>
              )}

              {gameState === "playing" && (
                <div className="space-y-2">
                  <div className="text-center p-4 bg-slate-700 rounded-lg">
                    <p className="text-slate-400 text-sm">Current Payout</p>
                    <p className="text-3xl font-bold text-yellow-500">${currentPayout.toFixed(2)}</p>
                    <p className="text-slate-400 text-xs mt-1">
                      {revealedTiles.length} / {25 - Number.parseInt(mines)} tiles revealed
                    </p>
                  </div>
                  <Button
                    className="w-full bg-yellow-500 hover:bg-yellow-600 text-slate-900 py-6 text-lg font-bold"
                    onClick={handleCashout}
                  >
                    Cash Out ${currentPayout.toFixed(2)}
                  </Button>
                </div>
              )}

              {(gameState === "won" || gameState === "lost") && (
                <div className="space-y-2">
                  <div className={`text-center p-4 rounded-lg ${gameState === "won" ? "bg-green-500/20 border border-green-500" : "bg-red-500/20 border border-red-500"}`}>
                    <p className="text-2xl font-bold">
                      {gameState === "won" ? "🎉 You Won!" : "💥 Game Over"}
                    </p>
                    <p className="text-xl mt-2">
                      {gameState === "won" ? `+$${currentPayout.toFixed(2)}` : `Lost $${currentBet.toFixed(2)}`}
                    </p>
                  </div>
                  <Button
                    className="w-full bg-blue-500 hover:bg-blue-600 text-white py-6 text-lg font-bold"
                    onClick={initializeGame}
                  >
                    Play Again
                  </Button>
                </div>
              )}
            </div>

            {/* Coin Balance */}
            <div className="bg-slate-800 rounded-xl p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Coins className="w-8 h-8 text-yellow-500" />
                  <div>
                    <p className="text-slate-400 text-sm">Your Balance</p>
                    <p className="text-2xl font-bold">${coins.toFixed(2)}</p>
                  </div>
                </div>
                {coins < 1 && (
                  <Button
                    onClick={claimCoins}
                    className="bg-yellow-500 hover:bg-yellow-600 text-slate-900 font-bold"
                  >
                    Claim $100
                  </Button>
                )}
              </div>
            </div>

            {/* Info Panel */}
            <div className="bg-slate-800 rounded-xl p-6">
              <h3 className="font-bold mb-3">AI Mode: {aiMode === "behavior" ? "Player Behavior" : aiMode === "strategy" ? "Optimal Strategy" : "Pattern Recognition"}</h3>
              <div className="text-sm text-slate-400 space-y-2 mb-4">
                {aiMode === "behavior" && (
                  <p>
                    🧠 AI silently learns your clicking patterns from all your games. 
                    After 10+ total moves, it trains a model to understand your strategy.
                  </p>
                )}
                {aiMode === "strategy" && (
                  <p>
                    📊 AI analyzes which moves lead to success vs failure across all games.
                    After 10+ moves, it builds a model of optimal play strategies.
                  </p>
                )}
                {aiMode === "pattern" && (
                  <p>
                    ✨ Bombs follow predictable patterns (diagonals, crosses, borders, etc.). 
                    AI learns these patterns across 10+ games to recognize the sequences.
                  </p>
                )}
              </div>
              
              <div className="border-t border-slate-700 pt-3 mt-3">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Total moves collected:</span>
                  <span className="font-bold text-green-400">{totalMoves}</span>
                </div>
                <div className="flex justify-between text-sm mt-2">
                  <span className="text-slate-400">Training status:</span>
                  <span className={`font-bold ${totalMoves >= 10 ? "text-green-400" : "text-yellow-400"}`}>
                    {totalMoves >= 10 ? "Training active" : `Need ${10 - totalMoves} more`}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}