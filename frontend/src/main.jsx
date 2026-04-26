import ReactDOM from 'react-dom/client';
import { Agentation } from 'agentation';
import App from './App.jsx';
import './app.css';

function Root() {
  return (
    <>
      <App />
      <Agentation />
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<Root />);
