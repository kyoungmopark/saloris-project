import axios from 'axios';
import React, { useState } from 'react';
import './cameraapp2.css';


function CameraView({ rtsp, index, onDrawingStart, onSpotSelected, drawingMode, parkingSpot, setParkingSpot }) {

  const handleCanvasClick = (e) => {
    if (drawingMode) {    
        const canvas = e.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const updatedParkingSpot = [...parkingSpot, { x, y }];
        setParkingSpot(updatedParkingSpot);
        console.log(`Clicked at: ${updatedParkingSpot.length}`); 
        if (updatedParkingSpot.length === 4) {
          onSpotSelected(index, updatedParkingSpot);
        }
    }
};

  return (
      <div className="camera-view" style={{ marginBottom: '50px' }}>
          <div style={{ position: 'relative' }}>
              <iframe
                  id={`camera-iframe-${index}`}
                  title={`Camera ${index + 1}`}
                  src={`http://localhost:8000/video/${index}`}
                  width="640"
                  height="480"
              ></iframe>
              <canvas
                  id={`overlay-canvas-${index}`}
                  width="640"
                  height="480"
                  style={{ position: 'absolute', top: 0, left: 0, pointerEvents: drawingMode ? 'auto' : 'none' }}
                  onClick={handleCanvasClick}
              ></canvas>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: '10px' }}>
              <button onClick={() => onDrawingStart(index)}>주차장 그리기</button>
              {drawingMode && (
                  <button style={{ marginLeft: '10px' }} onClick={onDrawingStart}>취소</button>
              )}
          </div>
      </div>
  );
}

export default function CameraApp2() {

    const rtspstr = [
      "rtsp://admin:saloris123!@192.168.0.106/profile2/media.smp",
      "rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp",
      "rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp",
      "rtsp://admin:saloris123!@192.168.0.107/profile2/media.smp",
      "rtsp://admin:saloris123!@192.168.0.111/profile2/media.smp",
      "rtsp://admin:saloris123!@192.168.0.124/profile2/media.smp"
    ];

    const [drawingCameraIndex, setDrawingCameraIndex] = useState(null);
    const [parkingSpot, setParkingSpot] = useState([]);
    const [parkingInfo, setParkingInfo] = useState([]);

    const fetchParkingInfo = async () => {
      try {
        const response = await axios.get('http://localhost:8000/get_parking_info');
        setParkingInfo(response.data.parkingInfo);
      } catch (error) {
        console.error(error);
      }
    };
    
    const handleDrawingStart = (cameraIndex) => {
      if (drawingCameraIndex === cameraIndex) {
          setDrawingCameraIndex(null);
          setParkingSpot([]);
      } else {
          setDrawingCameraIndex(cameraIndex);
          setParkingSpot([]);
      }
  };

  const handleSpotSelected = async (cameraIndex, spotCoordinates) => {
      try {
        const response = await axios.post('http://localhost:8000/add_parking_spot', {
          cameraIndex: cameraIndex,
          coordinates: spotCoordinates,
        });
          alert(response.data.message);
          setDrawingCameraIndex(null);
          setParkingSpot([]);
      } catch (error) {
          console.error(error);
      }
  };

  return (
    <div>
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
    <div className="camera-grid">
        {rtspstr.map((rtsp, index) => (
            <CameraView 
                key={index}
                rtsp={rtsp}
                index={index}
                onDrawingStart={handleDrawingStart}
                onSpotSelected={handleSpotSelected}
                drawingMode={drawingCameraIndex === index}
                parkingSpot={parkingSpot}
                setParkingSpot={setParkingSpot} 
            />
        ))}
    </div>
    </div>
);
}