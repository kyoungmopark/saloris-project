import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './cameraapp.css';

const CameraApp = () => {
  const [cameras, setCameras] = useState([]);
  const [rtspAddress, setRtspAddress] = useState('');
  const [currentCamera, setCurrentCamera] = useState(null);
  const [drawingMode, setDrawingMode] = useState(false);
  const [parkingSpot, setParkingSpot] = useState([]);
  const [parkingInfo, setParkingInfo] = useState([]);


  const handleCanvasClick = (e) => {
    if (drawingMode && parkingSpot.length < 4) {
      const cameraView = document.querySelector('.camera-view');
      const rect = cameraView.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setParkingSpot([...parkingSpot, { x, y }]);
    }
  };

  const toggleDrawingMode = () => {
    setDrawingMode(!drawingMode);
    setParkingSpot([]); // 모드 변경시 좌표 초기화
  };

  const fetchParkingInfo = async () => {
    try {
      const response = await axios.get('http://localhost:8000/get_parking_info');
      setParkingInfo(response.data.parkingInfo);
    } catch (error) {
      console.error(error);
    }
  };





  useEffect(() => {
    if (parkingSpot.length === 4) {
      // 서버에 POST 요청
      axios.post('http://localhost:8000/add_parking_spot', {cameraIndex: currentCamera, coordinates: parkingSpot })
        .then((response) => {
          alert(response.data.message);
        })
        .catch((error) => {
          console.error(error);
        });
      setDrawingMode(false); // 그리기 모드 해제
    }
  }, [parkingSpot, currentCamera]);

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const response = await axios.get('http://localhost:8000/get_cameras');
        setCameras(response.data.cameras);
      } catch (error) {
        console.error(error);
      }
    };

    fetchCameras();
  }, []);



  const isValidRtsp = (url) => {
    const pattern = new RegExp("^rtsp://");
    return pattern.test(url);
  }


  const addCamera = async () => {
    if (!isValidRtsp(rtspAddress)) {
      alert('Invalid RTSP URL!');
      return;
    }
  
    if (cameras.includes(rtspAddress)) {
      alert('This RTSP URL is already added!');
      return;
    }
  
    try {
      const response = await axios.post('http://localhost:8000/add_camera', { rtsp: rtspAddress });
      alert(response.data.message);
      setCameras([...cameras, rtspAddress]);
      setRtspAddress('');
    } catch (error) {
      console.error(error);
    }
  };

  const viewCamera = (index) => {
    setCurrentCamera(index);
    setDrawingMode(false); 
    setParkingSpot([]); 
  };

  return (
    <div className="container">
      <button onClick={fetchParkingInfo}>주차 현황 확인</button>
       <table>
        <thead>
          <tr>
            <th>카메라 번호</th>
            <th>주차장 번호</th>
            <th>주차 상태</th>
            <th>차량 ID</th>
          </tr>
        </thead>
        <tbody>
          {parkingInfo.map((spot, index) => (
            <tr key={index}>
              <td>{spot.cameraIndex}</td>
              <td>{spot.parkingindex}</td>
              <td>{spot.occupied ? "주차됨" : "비어있음"}</td>
              <td>{spot.vehicle_id !== null ? spot.vehicle_id : "N/A"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="add-camera-container">
        <input
          type="text"
          value={rtspAddress}
          onChange={(e) => setRtspAddress(e.target.value)}
        />
        <button onClick={addCamera}>카메라추가</button>
      </div>

      <div className="camera-buttons">
        {cameras.map((camera, index) => (
          <button key={index} onClick={() => viewCamera(index)}>카메라 {index}</button>
        ))}
      </div>

      {currentCamera !== null && (
  <div className="camera-view" onClick={(e) => {
    const iframe = document.getElementById('camera-iframe');
    const rect = iframe.getBoundingClientRect();
    const x = e.clientX - rect.left; // x position within the element
    const y = e.clientY - rect.top;  // y position within the element
    document.getElementById('coord').innerHTML = `x: ${x} y: ${y}`;
  }}>
    <h2>Camera {currentCamera}</h2> 
    <iframe
      id="camera-iframe"
      title={`Camera ${currentCamera + 1}`}
      src={`http://localhost:8000/video/${currentCamera}/640/480`}
      width="640"
      height="480"
    ></iframe>
    <canvas
      id="overlay-canvas"
      width="640"
      height="480"
      style={{ position: 'absolute', top: 0, left: 0 }}
      onClick={handleCanvasClick}
    ></canvas>
    <div id="coord"></div> {/* 마우스 좌표를 표시할 div */}
  </div> 
)}
      <div className="drawing-controls">
      {currentCamera !== null && (<button onClick={toggleDrawingMode}>{drawingMode ? '취소' : '주차선그리기'}</button>)}
      </div>
    </div>
  );
};

export default CameraApp;