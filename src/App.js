import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  CircularProgress,
  Card,
  CardContent,
  CardMedia,
  List,
  ListItem,
  ListItemText,
  CssBaseline,
  Divider,
  Snackbar,
  Alert,
  Grid,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    text: {
      primary: '#ffffff',
      secondary: '#cccccc',
    },
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    h3: {
      fontWeight: 700,
      fontSize: '3rem',
      color: '#ffffff',
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [heatmap, setHeatmap] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [snackbarOpen, setSnackbarOpen] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setHeatmap(null);
    setPredictions([]);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Upload failed');

      const data = await response.json();
      setHeatmap(data.heatmap);
      setPredictions(
        data.predictions.sort((a, b) => b.probability - a.probability)
      );
    } catch (error) {
      console.error('Error:', error);
      alert('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const generateReportText = () => {
    const top3 = predictions.slice(0, 3)
      .map((p, i) => `${i + 1}. ${p.label} - ${(p.probability * 100).toFixed(2)}%`)
      .join('\n');
    const nextSteps = [
      '• Consult a pulmonologist for further evaluation.',
      '• Consider additional imaging (e.g. CT scan) if advised.',
      '• Follow up with your primary care physician with this report.'
    ].join('\n');

    return `
Patient Chest X-Ray AI Report

Top 3 Most Likely Conditions:
${top3}

Recommended Next Steps:
${nextSteps}
`;
  };

  const copyReportToClipboard = () => {
    navigator.clipboard.writeText(generateReportText());
    setSnackbarOpen(true);
  };

  const downloadReport = () => {
    const blob = new Blob([generateReportText()], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'AI_Chest_XRay_Report.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ mt: 4, mb: 6 }}>
        <Typography variant="h3" align="center" gutterBottom>
          PulmonAI
        </Typography>
        <Typography variant="subtitle1" align="center" gutterBottom sx={{ color: 'text.secondary' }}>
          AI-Assisted Chest X-Ray Analysis
        </Typography>

        <Box display="flex" flexDirection="column" alignItems="center" gap={3} sx={{ mt: 4 }}>
          <Button
            variant="contained"
            component="label"
            startIcon={<CloudUploadIcon />}
            sx={{
              bgcolor: 'primary.main',
              color: '#fff',
              '&:hover': { bgcolor: 'primary.dark' },
            }}
          >
            Upload Chest X-Ray
            <input type="file" hidden onChange={handleFileChange} />
          </Button>

          {selectedFile && (
            <Typography variant="subtitle2" color="text.secondary">
              Selected: {selectedFile.name}
            </Typography>
          )}

          <Button
            variant="contained"
            color="secondary"
            disabled={!selectedFile || loading}
            onClick={handleSubmit}
            sx={{
              minWidth: 150,
            }}
          >
            {loading ? <CircularProgress size={24} /> : 'Analyze'}
          </Button>
        </Box>

        {heatmap && (
          <Card sx={{ mt: 5, borderRadius: 2, boxShadow: 3 }}>
            <CardMedia
              component="img"
              image={`data:image/png;base64,${heatmap}`}
              alt="Heatmap"
            />
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI-Generated Heatmap
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Highlighted regions represent areas of potential concern identified by the model.
              </Typography>
            </CardContent>
          </Card>
        )}

        {predictions.length > 0 && (
  <Box sx={{ mt: 5 }}>
    <Typography variant="h5" gutterBottom>
      Predictions
    </Typography>
    <Box display="flex" flexDirection="column" gap={2}>
      {predictions.map((pred, index) => {
        let riskLevel = '';
        let bgColor = '';
        let borderColor = '';
        let textColor = '';

        if (index < 4) {
          riskLevel = 'High risk level';
          bgColor = 'rgba(255, 0, 0, 0.1)';
          borderColor = '#ff4d4d';
          textColor = '#ff4d4d';
        } else if (index < 10) {
          riskLevel = 'Medium risk level';
          bgColor = 'rgba(255, 255, 0, 0.08)';
          borderColor = '#e6b800';
          textColor = '#e6b800';
        } else {
          riskLevel = 'Low risk level';
          bgColor = 'rgba(0, 255, 0, 0.08)';
          borderColor = '#4dff4d';
          textColor = '#4dff4d';
        }

        return (
          <Box
            key={index}
            sx={{
              backgroundColor: bgColor,
              borderLeft: `6px solid ${borderColor}`,
              borderRadius: 2,
              px: 3,
              py: 2,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              boxShadow: `0 0 10px ${borderColor}55`,
              transition: 'transform 0.2s',
              '&:hover': { transform: 'scale(1.02)' },
            }}
          >
            <Box>
              <Typography
                variant="h6"
                sx={{ color: textColor, fontWeight: 600, textTransform: 'capitalize' }}
              >
                {pred.label}
              </Typography>
              <Typography
                variant="body2"
                sx={{ color: textColor, mt: 0.5 }}
              >
                {riskLevel}
              </Typography>
            </Box>
            <Typography
              variant="h5"
              sx={{ color: textColor, fontWeight: 700 }}
            >
              {(pred.probability * 100).toFixed(2)}%
            </Typography>
          </Box>
        );
      })}
    </Box>
  </Box>
)}

{predictions.length > 0 && (
  <Card
    sx={{
      mt: 5,
      borderRadius: 2,
      boxShadow: 4,
      bgcolor: '#1e1e1e',
      border: '1px solid #333',
    }}
  >
    <CardContent>
      <Typography variant="h5" gutterBottom>
        Recommended Next Steps
      </Typography>
      <Divider sx={{ mb: 2, bgcolor: '#444' }} />
      <List>
        <ListItem>
          <ListItemText primary="Consult a pulmonologist for further evaluation." />
        </ListItem>
        <ListItem>
          <ListItemText primary="Consider additional imaging (e.g. CT scan) if advised." />
        </ListItem>
        <ListItem>
          <ListItemText primary="Follow up with your primary care physician with this report." />
        </ListItem>
      </List>
    </CardContent>
  </Card>
)}

{predictions.length > 0 && (
  <Card
    sx={{
      mt: 5,
      borderRadius: 2,
      boxShadow: 4,
      bgcolor: '#ffffff',
      color: '#000000',
    }}
  >
    <CardContent>
      <Typography variant="h5" gutterBottom>
        AI-Generated Report
      </Typography>
      <Divider sx={{ mb: 2, bgcolor: '#ccc' }} />

      <Typography variant="subtitle1" gutterBottom>
        <strong>Patient Chest X-Ray Report</strong>
      </Typography>

      <Typography variant="subtitle2" gutterBottom sx={{ color: 'text.secondary' }}>
        Top 3 Most Likely Conditions:
      </Typography>
      <List dense>
        {predictions.slice(0, 3).map((pred, idx) => (
          <ListItem key={idx}>
            <ListItemText
              primary={`${idx + 1}. ${pred.label}`}
              secondary={`Probability: ${(pred.probability * 100).toFixed(2)}%`}
            />
          </ListItem>
        ))}
      </List>

      <Typography variant="subtitle2" gutterBottom sx={{ mt: 2, color: 'text.secondary' }}>
        Recommended Next Steps:
      </Typography>
      <List dense>
        <ListItem>
          <ListItemText primary="Consult a pulmonologist for further evaluation." />
        </ListItem>
        <ListItem>
          <ListItemText primary="Consider additional imaging (e.g. CT scan) if advised." />
        </ListItem>
        <ListItem>
          <ListItemText primary="Follow up with your primary care physician with this report." />
        </ListItem>
      </List>

      <Box display="flex" gap={2} sx={{ mt: 3 }}>
        <Button
          variant="outlined"
          startIcon={<ContentCopyIcon />}
          onClick={copyReportToClipboard}
          sx={{
            borderColor: '#000',
            color: '#000',
            '&:hover': { borderColor: '#333', backgroundColor: '#f5f5f5' },
          }}
        >
          Copy Report
        </Button>
        <Button
          variant="contained"
          startIcon={<DownloadIcon />}
          onClick={downloadReport}
          sx={{
            backgroundColor: '#000',
            color: '#fff',
            '&:hover': { backgroundColor: '#333' },
          }}
        >
          Download Report
        </Button>
      </Box>
    </CardContent>
  </Card>
)}


        <Snackbar
          open={snackbarOpen}
          autoHideDuration={3000}
          onClose={() => setSnackbarOpen(false)}
        >
          <Alert severity="success" variant="filled" onClose={() => setSnackbarOpen(false)}>
            Report copied to clipboard!
          </Alert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  );
}

export default App;
