import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MapaBrasil from './Site/MapaBrasil';
import Dicas from './Site/Dicas';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MapaBrasil />} />
        <Route path="/Dicas" element={<Dicas />} />
      </Routes>
    </Router>
  );
}

export default App;
