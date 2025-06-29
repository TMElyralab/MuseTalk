"""
Avatar management service for MuseTalk WebSocket.
"""
import os
import glob
import pickle
import json
from typing import Optional, List, Dict, Any
import torch
import asyncio
from pathlib import Path

from models.session import AvatarInfo


class AvatarService:
    """Service for managing user avatars and base videos."""
    
    def __init__(self, 
                 avatars_dir: str = "../results/v15/avatars",
                 cache_size: int = 10):
        """
        Initialize avatar service.
        
        Args:
            avatars_dir: Directory containing user avatars
            cache_size: Maximum number of avatars to cache in memory
        """
        self.avatars_dir = Path(avatars_dir)
        self.cache_size = cache_size
        self.avatar_cache = {}  # user_id -> avatar data
        
        # Ensure avatars directory exists
        self.avatars_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_avatar(self, user_id: str) -> Optional[AvatarInfo]:
        """
        Load avatar for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Avatar information or None if not found
        """
        # Check cache first
        if user_id in self.avatar_cache:
            return self.avatar_cache[user_id]
        
        # Load from disk
        avatar_info = await self._load_avatar_from_disk(user_id)
        
        if avatar_info:
            # Add to cache (with LRU eviction)
            if len(self.avatar_cache) >= self.cache_size:
                # Remove oldest entry
                oldest = next(iter(self.avatar_cache))
                del self.avatar_cache[oldest]
            
            self.avatar_cache[user_id] = avatar_info
        
        return avatar_info
    
    async def _load_avatar_from_disk(self, user_id: str) -> Optional[AvatarInfo]:
        """Load avatar data from disk."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._load_avatar_sync,
            user_id
        )
    
    def _load_avatar_sync(self, user_id: str) -> Optional[AvatarInfo]:
        """Synchronously load avatar data."""
        avatar_path = self.avatars_dir / user_id
        
        # For POC, create mock data if not exists
        if not avatar_path.exists():
            return self._create_mock_avatar(user_id)
        
        try:
            # Load avatar info
            info_path = avatar_path / "avatar_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info_data = json.load(f)
            else:
                info_data = {}
            
            # Check for required files
            latents_path = avatar_path / "latents.pt"
            coords_path = avatar_path / "coords.pkl"
            mask_coords_path = avatar_path / "mask_coords.pkl"
            
            # Get available videos
            available_videos = self._get_available_videos(user_id)
            
            avatar_info = AvatarInfo(
                user_id=user_id,
                model_loaded=latents_path.exists(),
                available_videos=available_videos,
                avatar_path=str(avatar_path),
                latents_path=str(latents_path) if latents_path.exists() else None,
                coords_path=str(coords_path) if coords_path.exists() else None,
                mask_coords_path=str(mask_coords_path) if mask_coords_path.exists() else None
            )
            
            return avatar_info
            
        except Exception as e:
            print(f"Error loading avatar for {user_id}: {e}")
            return self._create_mock_avatar(user_id)
    
    def _get_available_videos(self, user_id: str) -> List[str]:
        """Get list of available base videos for user."""
        # Default video states
        video_states = []
        
        # Idle videos
        for i in range(7):
            video_states.append(f"idle_{i}")
        
        # Speaking videos
        for i in range(8):
            video_states.append(f"speaking_{i}")
        
        # Action videos
        video_states.extend(["action_1", "action_2"])
        
        return video_states
    
    def _create_mock_avatar(self, user_id: str) -> AvatarInfo:
        """Create mock avatar for testing."""
        return AvatarInfo(
            user_id=user_id,
            model_loaded=True,
            available_videos=self._get_available_videos(user_id),
            avatar_path=str(self.avatars_dir / user_id)
        )
    
    async def prepare_avatar(self, user_id: str, video_path: str) -> bool:
        """
        Prepare avatar from video (simplified version).
        
        In production, this would:
        1. Extract frames from video
        2. Detect face and landmarks
        3. Extract VAE latents
        4. Save avatar data
        
        Args:
            user_id: User identifier
            video_path: Path to source video
            
        Returns:
            True if successful
        """
        try:
            avatar_path = self.avatars_dir / user_id
            avatar_path.mkdir(parents=True, exist_ok=True)
            
            # For POC, just create mock files
            info_data = {
                "user_id": user_id,
                "source_video": video_path,
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            with open(avatar_path / "avatar_info.json", 'w') as f:
                json.dump(info_data, f)
            
            # Create mock latents
            mock_latents = torch.randn(100, 4, 32, 32)  # Mock VAE latents
            torch.save(mock_latents, avatar_path / "latents.pt")
            
            # Create mock coordinates
            mock_coords = [(100, 100, 400, 400)] * 100  # Mock face bboxes
            with open(avatar_path / "coords.pkl", 'wb') as f:
                pickle.dump(mock_coords, f)
            
            # Clear cache for this user
            if user_id in self.avatar_cache:
                del self.avatar_cache[user_id]
            
            return True
            
        except Exception as e:
            print(f"Error preparing avatar: {e}")
            return False
    
    def get_avatar_stats(self) -> Dict[str, Any]:
        """Get avatar service statistics."""
        return {
            "cache_size": len(self.avatar_cache),
            "max_cache_size": self.cache_size,
            "avatars_dir": str(self.avatars_dir),
            "cached_users": list(self.avatar_cache.keys())
        }
    
    def clear_cache(self, user_id: Optional[str] = None):
        """Clear avatar cache."""
        if user_id:
            if user_id in self.avatar_cache:
                del self.avatar_cache[user_id]
        else:
            self.avatar_cache.clear()
    
    async def load_avatar_latents(self, user_id: str) -> Optional[torch.Tensor]:
        """Load avatar latents for user."""
        avatar_info = await self.load_avatar(user_id)
        if not avatar_info or not avatar_info.latents_path:
            return None
        
        try:
            latents = torch.load(avatar_info.latents_path, map_location='cpu')
            return latents
        except Exception as e:
            print(f"Error loading latents: {e}")
            return None
    
    async def load_avatar_coords(self, user_id: str) -> Optional[List]:
        """Load avatar coordinates for user."""
        avatar_info = await self.load_avatar(user_id)
        if not avatar_info or not avatar_info.coords_path:
            return None
        
        try:
            with open(avatar_info.coords_path, 'rb') as f:
                coords = pickle.load(f)
            return coords
        except Exception as e:
            print(f"Error loading coordinates: {e}")
            return None