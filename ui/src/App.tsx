import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Index from './pages/Index'
import Backtest from './pages/Backtest'
import Study from './pages/Study'
import Trade from './pages/Trade'
import { TagsProvider } from './tags/TagsContext'

export default function App() {
  return (
    <TagsProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/backtest/:id" element={<Backtest />} />
          <Route path="/backtest/:id/trade/:tradeId" element={<Trade />} />
          <Route path="/study/:id" element={<Study />} />
        </Routes>
      </BrowserRouter>
    </TagsProvider>
  )
}
