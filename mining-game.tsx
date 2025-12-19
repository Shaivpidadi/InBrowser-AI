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
  const [predictions, setPredictions] = useState<{ tile: number; confidence: number }[]>([])
  const [aiInsight, setAiInsight] = useState<string>("")
  
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

  // MODE 1: Predict Player Behavior
  async function predictPlayerBehavior() {
    if (!behaviorModelRef.current || gameMovesRef.current.length < 10) {
      setAiInsight("Need at least 10 moves to predict behavior")
      return
    }

    const recentMoves = gameMovesRef.current.slice(-10).map(m => m.tileIndex)
    const inputTensor = tf.tensor2d([recentMoves], [1, 10])
    
    try {
      const prediction = behaviorModelRef.current.predict(inputTensor) as tf.Tensor
      const probs = await prediction.data()
      
      const sortedPredictions = Array.from(probs)
        .map((prob, idx) => ({ tile: idx, confidence: prob }))
        .filter(p => !revealedTiles.includes(p.tile))
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 5)
      
      setPredictions(sortedPredictions)
      setAiInsight(`AI predicts you'll click tile ${sortedPredictions[0].tile} next (${(sortedPredictions[0].confidence * 100).toFixed(1)}% confident)`)
      
      inputTensor.dispose()
      prediction.dispose()
    } catch (error) {
      console.error("Prediction error:", error)
    }
  }

  // MODE 2: Learn Optimal Strategy
  async function predictOptimalMove() {
    if (!strategyModelRef.current || revealedTiles.length === 0) {
      setAiInsight("AI needs revealed tiles to suggest optimal moves")
      return
    }

    const gameStateVector = extractGameStateVector(revealedTiles, grid)
    const input = [...gameStateVector, Number.parseInt(mines) / 25]
    const inputTensor = tf.tensor2d([input], [1, 26])
    
    try {
      const prediction = strategyModelRef.current.predict(inputTensor) as tf.Tensor
      const safetyScores = await prediction.data()
      
      const sortedMoves = Array.from(safetyScores)
        .map((score, idx) => ({ tile: idx, confidence: score }))
        .filter(p => !revealedTiles.includes(p.tile))
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 5)
      
      setPredictions(sortedMoves)
      
      const neighbors = getTileNeighbors(sortedMoves[0].tile)
      const revealedNeighbors = neighbors.filter(n => revealedTiles.includes(n))
      setAiInsight(`Safest move: tile ${sortedMoves[0].tile} (${(sortedMoves[0].confidence * 100).toFixed(1)}% safe, ${revealedNeighbors.length} neighbors revealed)`)
      
      inputTensor.dispose()
      prediction.dispose()
    } catch (error) {
      console.error("Strategy prediction error:", error)
    }
  }

  // MODE 3: Pattern-Based Bomb Placement
  function generatePatternBombs(): number[] {
    const numMines = Number.parseInt(mines)
    const round = patternStateRef.current.round
    
    // Define learnable patterns
    const patterns = [
      // Diagonal
      [0, 6, 12, 18, 24],
      // Cross
      [2, 10, 12, 14, 22],
      // Corners
      [0, 4, 20, 24, 12],
      // Border
      [0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21, 20, 15, 10, 5],
      // Checkerboard
      [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24],
      // Center cluster
      [6, 7, 8, 11, 12, 13, 16, 17, 18],
    ]
    
    const patternIndex = round % patterns.length
    const pattern = patterns[patternIndex]
    
    // Select mines from pattern
    const bombPositions: number[] = []
    const shuffled = [...pattern].sort(() => Math.random() - 0.5)
    
    for (let i = 0; i < Math.min(numMines, shuffled.length); i++) {
      bombPositions.push(shuffled[i])
    }
    
    // If pattern doesn't have enough positions, add random ones
    while (bombPositions.length < numMines) {
      const pos = Math.floor(Math.random() * 25)
      if (!bombPositions.includes(pos)) {
        bombPositions.push(pos)
      }
    }
    
    patternStateRef.current.lastPattern = bombPositions
    return bombPositions
  }

  async function predictPatternBombs() {
    if (!patternModelRef.current) {
      setAiInsight("Pattern model not ready")
      return
    }

    const round = patternStateRef.current.round
    const lastPattern = patternStateRef.current.lastPattern.length === 25 
      ? patternStateRef.current.lastPattern 
      : Array(25).fill(0)
    
    const input = [round / 100, ...lastPattern]
    const inputTensor = tf.tensor2d([input], [1, 26])
    
    try {
      const prediction = patternModelRef.current.predict(inputTensor) as tf.Tensor
      const bombProbs = await prediction.data()
      
      const sortedBombs = Array.from(bombProbs)
        .map((prob, idx) => ({ tile: idx, confidence: prob }))
        .filter(p => !revealedTiles.includes(p.tile))
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 5)
      
      setPredictions(sortedBombs)
      setAiInsight(`AI predicts bombs at: ${sortedBombs.slice(0, 3).map(b => b.tile).join(", ")} (confidence: ${(sortedBombs[0].confidence * 100).toFixed(1)}%)`)
      
      inputTensor.dispose()
      prediction.dispose()
    } catch (error) {
      console.error("Pattern prediction error:", error)
    }
  }

  // Training Functions
  async function trainBehaviorModel() {
    if (!behaviorModelRef.current) return
    
    const allMoves = await loadAllData(MOVES_STORE)
    if (allMoves.length < 20) {
      console.log("Need at least 20 moves to train behavior model")
      return
    }

    setTrainingProgress("Training behavior model...")
    
    try {
      const sequences: number[][] = []
      const labels: number[] = []
      
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
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            setTrainingProgress(`Behavior training: epoch ${epoch + 1}/10, loss: ${logs?.loss.toFixed(4)}`)
          }
        }
      })
      
      await behaviorModelRef.current.save("indexeddb://behavior-model")
      console.log("✅ Behavior model trained and saved")
      
      X.dispose()
      y.dispose()
      setTrainingProgress("")
    } catch (error) {
      console.error("Behavior training error:", error)
      setTrainingProgress("Training failed")
    }
  }

  async function trainStrategyModel() {
    if (!strategyModelRef.current) return
    
    const allMoves = await loadAllData(STRATEGY_STORE)
    if (allMoves.length < 50) {
      console.log("Need at least 50 strategy samples to train")
      return
    }

    setTrainingProgress("Training strategy model...")
    
    try {
      const X_data: number[][] = []
      const y_data: number[][] = []
      
      for (const move of allMoves) {
        X_data.push([...move.gameState, move.minesCount / 25])
        
        // Create target: 1 for the tile clicked (if successful), 0 otherwise
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
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            setTrainingProgress(`Strategy training: epoch ${epoch + 1}/15, accuracy: ${((logs?.acc || 0) * 100).toFixed(1)}%`)
          }
        }
      })
      
      await strategyModelRef.current.save("indexeddb://strategy-model")
      console.log("✅ Strategy model trained and saved")
      
      X.dispose()
      y.dispose()
      setTrainingProgress("")
    } catch (error) {
      console.error("Strategy training error:", error)
      setTrainingProgress("Training failed")
    }
  }

  async function trainPatternModel() {
    if (!patternModelRef.current) return
    
    const allPatterns = await loadAllData(PATTERN_STORE)
    if (allPatterns.length < 10) {
      console.log("Need at least 10 pattern rounds to train")
      return
    }

    setTrainingProgress("Training pattern model...")
    
    try {
      const X_data: number[][] = []
      const y_data: number[][] = []
      
      for (let i = 0; i < allPatterns.length - 1; i++) {
        const current = allPatterns[i]
        const next = allPatterns[i + 1]
        
        const lastPattern = current.lastPattern.length === 25 ? current.lastPattern : Array(25).fill(0)
        X_data.push([current.round / 100, ...lastPattern])
        
        // Create target: 1 where bombs are, 0 elsewhere
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
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            setTrainingProgress(`Pattern training: epoch ${epoch + 1}/20, loss: ${logs?.loss.toFixed(4)}`)
          }
        }
      })
      
      await patternModelRef.current.save("indexeddb://pattern-model")
      console.log("✅ Pattern model trained and saved")
      
      X.dispose()
      y.dispose()
      setTrainingProgress("")
    } catch (error) {
      console.error("Pattern training error:", error)
      setTrainingProgress("Training failed")
    }
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
    
    // Use pattern-based placement if in pattern mode
    if (aiMode === "pattern") {
      bombPositions = generatePatternBombs()
      patternStateRef.current.round++
      savePatternState(patternStateRef.current)
    } else {
      // Random placement for behavior and strategy modes
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
    setPredictions([])
    setAiInsight("")
    currentGameStartRef.current = Date.now()
    gameMovesRef.current = []
    
    // Make initial prediction based on mode
    if (aiMode === "pattern") {
      setTimeout(() => predictPatternBombs(), 500)
    }
  }

  const handleTileClick = async (index: number) => {
    if (gameState !== "playing" || revealedTiles.includes(index)) return

    const clickTime = Date.now()
    const newRevealedTiles = [...revealedTiles, index]
    setRevealedTiles(newRevealedTiles)

    const wasSuccessful = grid[index] === "gem"
    
    // Record move for training
    const move: GameMove = {
      tileIndex: index,
      timestamp: clickTime - currentGameStartRef.current,
      gameState: extractGameStateVector(revealedTiles, grid),
      wasSuccessful,
      minesCount: Number.parseInt(mines),
      revealedCount: revealedTiles.length
    }
    
    gameMovesRef.current.push(move)
    
    // Save to appropriate store
    if (aiMode === "behavior") {
      await saveData(MOVES_STORE, move)
    } else if (aiMode === "strategy") {
      await saveData(STRATEGY_STORE, move)
    }

    if (grid[index] === "bomb") {
      setGameState("lost")
      setRevealedTiles([...Array(25).keys()])
      setCurrentPayout(0)
      setAiInsight("💥 Hit a bomb! Game over.")
      
      // Train models after game ends
      setTimeout(() => {
        if (aiMode === "behavior") trainBehaviorModel()
        else if (aiMode === "strategy") trainStrategyModel()
        else if (aiMode === "pattern") trainPatternModel()
      }, 1000)
    } else {
      const safeReveals = newRevealedTiles.length
      const newPayout = calculatePayout(currentBet, Number.parseInt(mines), safeReveals)
      setCurrentPayout(newPayout)

      // Make predictions based on mode
      if (aiMode === "behavior") {
        predictPlayerBehavior()
      } else if (aiMode === "strategy") {
        predictOptimalMove()
      }

      if (safeReveals === 25 - Number.parseInt(mines)) {
        setGameState("won")
        setRevealedTiles([...Array(25).keys()])
        setCoins(prevCoins => prevCoins + newPayout)
        setAiInsight("🎉 You won! Perfect game!")
        
        // Train models after winning
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
      setAiInsight(`💰 Cashed out $${currentPayout.toFixed(2)}!`)
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

  const getPredictionColor = (tileIndex: number): string => {
    const pred = predictions.find(p => p.tile === tileIndex)
    if (!pred) return ""
    
    const confidence = pred.confidence
    if (confidence > 0.7) return "ring-2 ring-red-500 ring-offset-2"
    if (confidence > 0.5) return "ring-2 ring-yellow-500 ring-offset-2"
    if (confidence > 0.3) return "ring-2 ring-blue-500 ring-offset-2"
    return ""
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">AI Mining Game</h1>
          <p className="text-slate-400">Three AI modes: Learn player behavior, optimal strategy, or pattern recognition</p>
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
                AI learns YOUR clicking patterns and predicts where you'll click next
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
                AI analyzes game state and suggests the statistically safest moves
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
                Bombs follow learnable patterns. AI predicts bomb locations!
              </p>
            </button>
          </div>

          {/* AI Insight */}
          {aiInsight && (
            <div className="mt-4 p-3 bg-slate-700/50 rounded-lg">
              <p className="text-sm text-slate-300">{aiInsight}</p>
            </div>
          )}

          {/* Training Progress */}
          {trainingProgress && (
            <div className="mt-4 p-3 bg-blue-500/20 border border-blue-500 rounded-lg">
              <p className="text-sm text-blue-300">{trainingProgress}</p>
            </div>
          )}
        </div>

        {/* Main Game Area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Game Grid */}
          <div className="bg-slate-800 rounded-xl p-6">
            <div className="grid grid-cols-5 gap-2">
              {grid.map((tile, index) => (
                <motion.button
                  key={index}
                  className={`aspect-square rounded-lg p-4 flex items-center justify-center relative
                    ${revealedTiles.includes(index) ? "bg-slate-700" : "bg-slate-700 hover:bg-slate-600"}
                    ${getPredictionColor(index)}`}
                  onClick={() => handleTileClick(index)}
                  whileHover={{ scale: gameState === "playing" ? 1.05 : 1 }}
                  whileTap={{ scale: gameState === "playing" ? 0.95 : 1 }}
                  disabled={gameState !== "playing"}
                >
                  {/* Prediction indicator */}
                  {predictions.find(p => p.tile === index) && !revealedTiles.includes(index) && (
                    <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                  )}

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

            {/* Predictions Display */}
            {predictions.length > 0 && gameState === "playing" && (
              <div className="mt-4 p-3 bg-slate-700/50 rounded-lg">
                <p className="text-xs font-bold mb-2">AI Predictions:</p>
                <div className="flex gap-2 flex-wrap">
                  {predictions.slice(0, 5).map((pred, idx) => (
                    <span
                      key={pred.tile}
                      className="text-xs px-2 py-1 bg-slate-600 rounded"
                    >
                      #{pred.tile} ({(pred.confidence * 100).toFixed(0)}%)
                    </span>
                  ))}
                </div>
              </div>
            )}
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
              <h3 className="font-bold mb-2">How It Works:</h3>
              <div className="text-sm text-slate-400 space-y-2">
                {aiMode === "behavior" && (
                  <p>
                    🧠 The AI tracks your clicking patterns and learns to predict where you'll click next. 
                    The more you play, the better it gets at understanding your strategy!
                  </p>
                )}
                {aiMode === "strategy" && (
                  <p>
                    📊 The AI analyzes revealed tiles and recommends statistically safe moves. 
                    It learns from successful games to build an optimal strategy.
                  </p>
                )}
                {aiMode === "pattern" && (
                  <p>
                    ✨ Bombs follow predictable patterns (diagonals, crosses, borders, etc.). 
                    The AI learns these patterns and predicts bomb locations. This is the only mode where AI can actually find bombs!
                  </p>
                )}
                <p className="text-xs text-slate-500 mt-4">
                  💡 Tip: Highlighted tiles show AI predictions. Red = high confidence, Yellow = medium, Blue = low
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}