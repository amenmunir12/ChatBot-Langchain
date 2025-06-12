import React, { useState } from 'react';
import axios from 'axios';
import { FaPaperPlane, FaUpload } from 'react-icons/fa';

function ChatWithUpload() {
  const [messages, setMessages] = useState([
    { from: 'bot', text: 'Hello! How can I assist you today? Please upload a file' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setMessages(prev => [...prev, { from: 'user', text: input }]);
    setLoading(true);
    setInput('');

    try {
      const res = await fetch('http://localhost:8000/api/ask/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      });

      if (!res.ok) throw new Error('Server error');
      const data = await res.json();
      setMessages(prev => [...prev, { from: 'bot', text: data.response }]);
    } catch (err) {
      alert('Server error while sending message');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      alert('Please select a file first');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('http://localhost:8000/api/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: e => {
          const progress = Math.round((e.loaded * 100) / e.total);
          setUploadProgress(progress);
        }
      });

      alert(res.data.message || 'File uploaded successfully!');
      setFile(null);
    } catch (err) {
      alert(err.response?.data?.error || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>ChatBot</h2>

      <div style={styles.chatBox}>
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              ...styles.msg,
              alignSelf: m.from === 'user' ? 'flex-end' : 'flex-start',
              backgroundColor: m.from === 'user' ? '#4e8cff' : '#2a2f4c',
              color: m.from === 'user' ? '#fff' : '#ccc',
              borderTopRightRadius: m.from === 'user' ? '0' : '20px',
              borderTopLeftRadius: m.from === 'user' ? '20px' : '0'
            }}
          >
            {m.text}
          </div>
        ))}
        {loading && <em style={styles.loading}>Assistant is typing...</em>}
      </div>

      <div style={styles.inputArea}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
          style={styles.input}
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          style={styles.btn}
          disabled={loading}
        >
          <FaPaperPlane style={styles.icon} />
        </button>
      </div>

      <div style={styles.uploadArea}>
        <input
          type="file"
          onChange={e => setFile(e.target.files[0])}
          disabled={uploading}
          style={styles.fileInput}
        />
        <button
          onClick={handleFileUpload}
          style={styles.btn}
          disabled={uploading}
        >
          <FaUpload style={styles.icon} />
          {uploading ? 'Uploading...' : 'Upload File'}
        </button>
      </div>

      {uploading && (
        <div style={styles.progressBar}>
          <div style={{ ...styles.progress, width: `${uploadProgress}%` }} />
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '800px',
    margin: '40px auto',
    padding: '24px',
    borderRadius: '20px',
    background: '#1e1e2f',
    boxShadow: '0 10px 25px rgba(0,0,0,0.3)',
    fontFamily: 'Inter, sans-serif',
    height: '90vh',
    display: 'flex',
    flexDirection: 'column',
    color: '#fff'
  },
  title: {
    textAlign: 'center',
    marginBottom: '20px',
    fontSize: '26px',
    fontWeight: '600',
    color: '#eee'
  },
  chatBox: {
    flex: 1,
    padding: '16px',
    overflowY: 'auto',
    background: '#23243a',
    borderRadius: '16px',
    border: '1px solid #444',
    marginBottom: '16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px'
  },
  msg: {
    padding: '12px 18px',
    borderRadius: '20px',
    marginBottom: '4px',
    maxWidth: '75%',
    fontSize: '15px',
    lineHeight: '1.4',
    wordBreak: 'break-word',
    boxShadow: '0 2px 5px rgba(0,0,0,0.3)'
  },
  loading: {
    fontStyle: 'italic',
    color: '#aaa',
    alignSelf: 'center',
    fontSize: '14px'
  },
  inputArea: {
    display: 'flex',
    gap: '10px',
    marginBottom: '12px'
  },
  input: {
    flex: 1,
    padding: '12px 16px',
    borderRadius: '30px',
    border: '1px solid #555',
    fontSize: '15px',
    backgroundColor: '#2c2c3e',
    color: '#fff',
    outline: 'none'
  },
  btn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 20px',
    borderRadius: '30px',
    border: 'none',
    background: 'linear-gradient(135deg, #4e8cff, #3162d1)',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: '600',
    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
    boxShadow: '0 4px 12px rgba(0,0,0,0.4)'
  },
  icon: {
    fontSize: '1.1em'
  },
  uploadArea: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '16px'
  },
  fileInput: {
    color: '#ccc'
  },
  progressBar: {
    height: '8px',
    background: '#444',
    borderRadius: '4px',
    overflow: 'hidden',
    marginTop: '4px'
  },
  progress: {
    height: '100%',
    background: '#4e8cff',
    transition: 'width 0.4s ease'
  }
};

export default ChatWithUpload;
