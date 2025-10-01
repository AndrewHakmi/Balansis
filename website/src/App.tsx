import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import Home from './pages/Home';
import { Docs } from './pages/Docs';
import { Examples } from './pages/Examples';
import { Theory } from './pages/Theory';
import { Community } from './pages/Community';
import { Download } from './pages/Download';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-primary-900 via-primary-800 to-primary-700">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/docs" element={<Docs />} />
            <Route path="/examples" element={<Examples />} />
            <Route path="/theory" element={<Theory />} />
            <Route path="/community" element={<Community />} />
            <Route path="/download" element={<Download />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
