"""
Session management utilities for ElectroML.
Handles user sessions and data storage.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions and associated data."""
    
    def __init__(self):
        """Initialize session storage."""
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'data': None,
            'features': None,
            'models': {},
            'preprocessing_config': {},
            'training_config': {}
        }
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, key: str, value: Any) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            key: Data key to update
            value: New value
            
        Returns:
            True if successful, False otherwise
        """
        if session_id in self._sessions:
            self._sessions[session_id][key] = value
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions."""
        return self._sessions
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Remove sessions older than specified hours.
        
        Args:
            max_age_hours: Maximum age in hours
        """
        current_time = datetime.now()
        sessions_to_delete = []
        
        for session_id, session_data in self._sessions.items():
            created_at = datetime.fromisoformat(session_data['created_at'])
            age_hours = (current_time - created_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            self.delete_session(session_id)
        
        if sessions_to_delete:
            logger.info(f"Cleaned up {len(sessions_to_delete)} old sessions")


# Global session manager instance
session_manager = SessionManager()