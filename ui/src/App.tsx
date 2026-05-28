import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Index from './pages/Index'
import Backtest from './pages/Backtest'
import Study from './pages/Study'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Index />} />
        <Route path="/backtest/:id" element={<Backtest />} />
        <Route path="/study/:id" element={<Study />} />
      </Routes>
    </BrowserRouter>
  )
}
