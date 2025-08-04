"""
Database module for storing lap times and sector data
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LapTimeDatabase:
    def __init__(self, db_path: str = "laptimes.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date_utc TEXT NOT NULL,
                    track_name TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create laps table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS laps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    lap_number INTEGER NOT NULL,
                    lap_time REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id),
                    UNIQUE(session_id, lap_number)
                )
            """
            )

            # Create sectors table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lap_id INTEGER NOT NULL,
                    sector_number INTEGER NOT NULL,
                    sector_time REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (lap_id) REFERENCES laps (id),
                    UNIQUE(lap_id, sector_number)
                )
            """
            )

            # Create indexes for better performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_track_date ON sessions (track_name, session_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_laps_session ON laps (session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sectors_lap ON sectors (lap_id)"
            )

            conn.commit()
            logger.info("Database initialized successfully")

    def get_or_create_session(
        self, date_utc: str, track_name: str, session_date: str
    ) -> int:
        """Get existing session or create new one"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Try to find existing session
            cursor.execute(
                """
                SELECT id FROM sessions
                WHERE date_utc = ? AND track_name = ? AND session_date = ?
            """,
                (date_utc, track_name, session_date),
            )

            result = cursor.fetchone()
            if result:
                logger.info(f"Found existing session ID: {result[0]}")
                return result[0]

            # Create new session
            cursor.execute(
                """
                INSERT INTO sessions (date_utc, track_name, session_date)
                VALUES (?, ?, ?)
            """,
                (date_utc, track_name, session_date),
            )

            session_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created new session ID: {session_id}")
            return session_id

    def store_lap_with_sectors(
        self,
        session_id: int,
        lap_number: int,
        lap_time: float,
        sector_times: List[float],
    ) -> bool:
        """Store lap time and its sectors (avoid duplicates)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                # Check if lap already exists
                cursor.execute(
                    "SELECT id FROM laps WHERE session_id = ? AND lap_number = ?",
                    (session_id, lap_number),
                )
                existing_lap = cursor.fetchone()

                if existing_lap:
                    logger.debug(
                        f"Lap {lap_number} already exists for session {session_id}"
                    )
                    return False

                # Insert lap
                cursor.execute(
                    """
                    INSERT INTO laps (session_id, lap_number, lap_time)
                    VALUES (?, ?, ?)
                """,
                    (session_id, lap_number, lap_time),
                )

                lap_id = cursor.lastrowid

                # Insert sectors
                for sector_num, sector_time in enumerate(sector_times, 1):
                    if sector_time > 0:  # Only store valid sector times
                        cursor.execute(
                            """
                            INSERT INTO sectors (lap_id, sector_number, sector_time)
                            VALUES (?, ?, ?)
                        """,
                            (lap_id, sector_num, sector_time),
                        )

                conn.commit()
                logger.info(
                    f"Stored lap {lap_number}: {lap_time}s with {len(sector_times)} sectors"
                )
                return True

            except sqlite3.IntegrityError as e:
                logger.warning(f"Lap {lap_number} already exists: {e}")
                return False

    def store_lap(self, session_id: int, lap_number: int, lap_time: float) -> bool:
        """Store individual lap time (avoid duplicates)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                # Check if lap already exists
                cursor.execute(
                    "SELECT id FROM laps WHERE session_id = ? AND lap_number = ?",
                    (session_id, lap_number),
                )
                existing_lap = cursor.fetchone()

                if existing_lap:
                    logger.debug(
                        f"Lap {lap_number} already exists for session {session_id}"
                    )
                    return False

                # Insert lap
                cursor.execute(
                    """
                    INSERT INTO laps (session_id, lap_number, lap_time)
                    VALUES (?, ?, ?)
                """,
                    (session_id, lap_number, lap_time),
                )

                conn.commit()
                logger.info(f"Stored lap {lap_number}: {lap_time}s")
                return True

            except sqlite3.IntegrityError as e:
                logger.warning(f"Lap {lap_number} already exists: {e}")
                return False

    def store_sector(
        self, session_id: int, lap_number: int, sector_number: int, sector_time: float
    ) -> bool:
        """Store individual sector time"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                # Get lap_id
                cursor.execute(
                    "SELECT id FROM laps WHERE session_id = ? AND lap_number = ?",
                    (session_id, lap_number),
                )
                lap_result = cursor.fetchone()

                if not lap_result:
                    logger.warning(
                        f"Lap {lap_number} not found for session {session_id}"
                    )
                    return False

                lap_id = lap_result[0]

                # Check if sector already exists
                cursor.execute(
                    "SELECT id FROM sectors WHERE lap_id = ? AND sector_number = ?",
                    (lap_id, sector_number),
                )
                existing_sector = cursor.fetchone()

                if existing_sector:
                    logger.debug(
                        f"Sector {sector_number} already exists for lap {lap_number}"
                    )
                    return False

                # Insert sector
                cursor.execute(
                    """
                    INSERT INTO sectors (lap_id, sector_number, sector_time)
                    VALUES (?, ?, ?)
                """,
                    (lap_id, sector_number, sector_time),
                )

                conn.commit()
                logger.info(
                    f"Stored sector {sector_number} for lap {lap_number}: {sector_time}s"
                )
                return True

            except sqlite3.IntegrityError as e:
                logger.warning(
                    f"Sector {sector_number} for lap {lap_number} already exists: {e}"
                )
                return False

    def get_fastest_lap_in_session(
        self, session_id: int
    ) -> Optional[Tuple[int, float]]:
        """Get the fastest lap number and time in a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT lap_number, lap_time
                FROM laps
                WHERE session_id = ?
                ORDER BY lap_time ASC
                LIMIT 1
            """,
                (session_id,),
            )
            return cursor.fetchone()

    def get_fastest_lap_overall(
        self, track_name: str
    ) -> Optional[Tuple[int, float, str]]:
        """Get the fastest lap overall for a track"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT l.lap_number, l.lap_time, s.session_date
                FROM laps l
                JOIN sessions s ON l.session_id = s.id
                WHERE s.track_name = ?
                ORDER BY l.lap_time ASC
                LIMIT 1
            """,
                (track_name,),
            )
            return cursor.fetchone()

    def get_fastest_sector_in_session(
        self, session_id: int, sector_number: int
    ) -> Optional[float]:
        """Get fastest sector time in a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT MIN(sect.sector_time)
                FROM sectors sect
                JOIN laps l ON sect.lap_id = l.id
                WHERE l.session_id = ? AND sect.sector_number = ?
            """,
                (session_id, sector_number),
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_fastest_sector_overall(
        self, track_name: str, sector_number: int
    ) -> Optional[float]:
        """Get fastest sector time overall for a track"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT MIN(sect.sector_time)
                FROM sectors sect
                JOIN laps l ON sect.lap_id = l.id
                JOIN sessions s ON l.session_id = s.id
                WHERE s.track_name = ? AND sect.sector_number = ?
            """,
                (track_name, sector_number),
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_session_laps(self, session_id: int) -> List[Tuple[int, float, List[float]]]:
        """Get all laps with sectors for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get laps
            cursor.execute(
                """
                SELECT id, lap_number, lap_time
                FROM laps
                WHERE session_id = ?
                ORDER BY lap_number
            """,
                (session_id,),
            )

            laps = []
            for lap_id, lap_number, lap_time in cursor.fetchall():
                # Get sectors for this lap
                cursor.execute(
                    """
                    SELECT sector_time
                    FROM sectors
                    WHERE lap_id = ?
                    ORDER BY sector_number
                """,
                    (lap_id,),
                )

                sectors = [row[0] for row in cursor.fetchall()]
                laps.append((lap_number, lap_time, sectors))

            return laps
