from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, QLineF
from PySide6.QtGui import QPainterPath, QPen, QPolygonF, QColor, QBrush
import math
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsEllipseItem
import shiboken6

from nodes.base.link_model import LinkModel, LinkState


# Color scheme for different data types
DATA_TYPE_COLORS = {
    "numeric": QColor(70, 130, 200),      # 🔵 Blue
    "categorical": QColor(70, 180, 120),  # 🟢 Green  
    "target": QColor(180, 100, 200),      # 🟣 Purple
    "tensor": QColor(220, 160, 60),       # 🟠 Orange
    "metrics": QColor(80, 200, 200),      # 🔵 Cyan
    "text": QColor(200, 150, 100),        # Brown
    "datetime": QColor(150, 100, 200),    # Violet
    "any": QColor(150, 160, 170),         # Gray
}

# Colors for link states
STATE_COLORS = {
    LinkState.NORMAL: None,  # Use data type color
    LinkState.WARNING: QColor(230, 180, 50),    # Yellow
    LinkState.ERROR: QColor(220, 80, 80),       # Red
    LinkState.DISABLED: QColor(100, 100, 100),  # Gray
    LinkState.LOCKED: QColor(100, 140, 180),    # Steel blue
    LinkState.EXECUTING: QColor(100, 200, 255), # Bright blue
    LinkState.COMPLETED: QColor(100, 200, 100), # Green
}


class EdgeItem(QGraphicsPathItem):
    """
    Visual representation of a connecting link between nodes.
    
    Features:
    - Bezier curve rendering
    - Color-coded by data type
    - Thickness based on column count
    - Visual states (normal, warning, error, etc.)
    - Arrow indicator for direction
    - Glow effect for active/executing state
    - Automatic position update when nodes move
    """
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._start = QPointF()
        self._end = QPointF()
        self._data_type = "any"
        self._state = LinkState.NORMAL
        self._column_count = 0
        self._is_temporary = False  # For drag preview
        
        # Port references for dynamic positioning
        self._source_port = None
        self._target_port = None
        
        # Visual settings
        self._base_width = 2
        self._arrow_size = 10
        self._glow_radius = 4
        
        # Link data
        self.columns: list[str] = []
        self.link: LinkModel | None = None
        
        # Set default pen
        self._update_pen()
        self.setZValue(-1)  # Draw behind nodes
        self.setAcceptHoverEvents(True)
        
        # Arrow head (drawn separately for better control)
        self._arrow_item = None
    
    def set_temporary(self, is_temp: bool) -> None:
        """Mark as temporary edge (used during drag)."""
        self._is_temporary = is_temp
        self._update_pen()
    
    def set_points(self, start: QPointF, end: QPointF) -> None:
        """Set the start and end points of the link."""
        self._start = start
        self._end = end
        self._update_path()
        try:
            sc = self.scene()
            if sc and shiboken6.isValid(sc) and hasattr(sc, "ensure_rect_in_scene"):
                sc.ensure_rect_in_scene(self.path().boundingRect())
        except Exception:
            pass
    
    def set_columns(self, columns: list[str]) -> None:
        """Set the columns passing through this link."""
        self.columns = columns
        self._column_count = len(columns)
        self._update_pen()
        if self.link:
            self.link.columns_passed = columns
    
    def set_data_type(self, data_type: str) -> None:
        """Set the data type for color coding."""
        self._data_type = data_type
        self._update_pen()
        if self.link:
            self.link.data_type = data_type
    
    def set_state(self, state: LinkState) -> None:
        """Set the visual state of the link."""
        self._state = state
        self._update_pen()
        if self.link:
            self.link.state = state
    
    def set_link_model(self, link: LinkModel) -> None:
        """Set the full link model."""
        self.link = link
        self.columns = link.columns_passed
        self._column_count = link.enabled_columns
        self._data_type = link.data_type
        self._state = link.state
        self._update_pen()
    
    def connect_ports(self, source_port, target_port) -> None:
        """
        Connect this edge to source and target ports.
        The edge will automatically update its position when ports move.
        """
        # Unregister from previous ports (if reconnecting).
        try:
            if self._source_port and shiboken6.isValid(self._source_port) and hasattr(self._source_port, "unregister_edge"):
                self._source_port.unregister_edge(self)
            if self._target_port and shiboken6.isValid(self._target_port) and hasattr(self._target_port, "unregister_edge"):
                self._target_port.unregister_edge(self)
        except Exception:
            pass

        self._source_port = source_port
        self._target_port = target_port

        try:
            if self._source_port and shiboken6.isValid(self._source_port) and hasattr(self._source_port, "register_edge"):
                self._source_port.register_edge(self)
            if self._target_port and shiboken6.isValid(self._target_port) and hasattr(self._target_port, "register_edge"):
                self._target_port.register_edge(self)
        except Exception:
            pass
        self.update_position()

    def disconnect_ports(self) -> None:
        """Detach this edge from tracked port registrations."""
        try:
            if self._source_port and shiboken6.isValid(self._source_port) and hasattr(self._source_port, "unregister_edge"):
                self._source_port.unregister_edge(self)
        except Exception:
            pass
        try:
            if self._target_port and shiboken6.isValid(self._target_port) and hasattr(self._target_port, "unregister_edge"):
                self._target_port.unregister_edge(self)
        except Exception:
            pass
        self._source_port = None
        self._target_port = None
    
    def update_position(self) -> None:
        """Update edge position based on connected ports."""
        if self._source_port and self._target_port:
            # Check if Qt objects are still valid before accessing
            if not shiboken6.isValid(self._source_port) or not shiboken6.isValid(self._target_port):
                return
            try:
                start = self._source_port.scenePos()
                end = self._target_port.scenePos()
                self.set_points(start, end)
            except RuntimeError:
                # Object may have been deleted
                pass
    
    def get_source_port(self):
        """Get the source port reference."""
        return self._source_port
    
    def get_target_port(self):
        """Get the target port reference."""
        return self._target_port
    
    def is_connected_to(self, port) -> bool:
        """Check if this edge is connected to a specific port."""
        return self._source_port == port or self._target_port == port
    
    def is_connected_to_any(self, ports: list) -> bool:
        """Check if this edge is connected to any port in the list."""
        return self._source_port in ports or self._target_port in ports
    
    def _get_color(self) -> QColor:
        """Get the appropriate color based on state and data type."""
        # State colors override data type colors
        state_color = STATE_COLORS.get(self._state)
        if state_color is not None:
            return state_color
        # Otherwise use data type color
        return DATA_TYPE_COLORS.get(self._data_type, DATA_TYPE_COLORS["any"])
    
    def _get_width(self) -> int:
        """Calculate line width based on column count."""
        # Base width + scaling for column count (max 10px)
        return self._base_width + min(8, max(0, self._column_count - 1))
    
    def _update_pen(self) -> None:
        """Update the pen based on current state."""
        color = self._get_color()
        width = self._get_width()
        
        pen = QPen(color, width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        
        # Dashed line for disabled or temporary
        if self._is_temporary:
            pen.setStyle(Qt.DashLine)
            pen.setColor(QColor(color.red(), color.green(), color.blue(), 150))
        elif self._state == LinkState.DISABLED:
            pen.setStyle(Qt.DashLine)
        elif self._state == LinkState.ERROR:
            pen.setStyle(Qt.DashDotLine)
        elif self._state == LinkState.LOCKED:
            pen.setStyle(Qt.DotLine)
        
        self.setPen(pen)
    
    def _update_path(self) -> None:
        """Update the Bezier curve path."""
        path = QPainterPath(self._start)
        
        # Calculate control points for smooth Bezier curve
        dx = abs(self._end.x() - self._start.x())
        dy = abs(self._end.y() - self._start.y())
        
        # Offset for control points (more horizontal = smoother curve)
        offset = max(50, dx * 0.5)
        
        c1 = QPointF(self._start.x() + offset, self._start.y())
        c2 = QPointF(self._end.x() - offset, self._end.y())
        
        path.cubicTo(c1, c2, self._end)
        self.setPath(path)
    
    def paint(self, painter, option, widget=None) -> None:
        """Custom paint for the edge with arrow and optional effects."""
        # Draw glow effect for executing state
        if self._state == LinkState.EXECUTING:
            glow_pen = QPen(self._get_color(), self._get_width() + self._glow_radius)
            glow_pen.setCapStyle(Qt.RoundCap)
            color = glow_pen.color()
            color.setAlpha(80)
            glow_pen.setColor(color)
            painter.setPen(glow_pen)
            painter.drawPath(self.path())
        
        # Draw main path
        super().paint(painter, option, widget)
        
        # Draw arrow at the end
        self._draw_arrow(painter)
        
        # Draw lock icon if locked
        if self._state == LinkState.LOCKED:
            self._draw_lock_icon(painter)
    
    def _draw_arrow(self, painter) -> None:
        """Draw directional arrow at the end of the link."""
        if self._start == self._end:
            return
        
        # Get direction at end of curve
        path = self.path()
        if path.isEmpty():
            return
        
        # Calculate arrow position slightly before the end
        t = 0.95  # Position along the path (0-1)
        arrow_pos = path.pointAtPercent(t)
        end_pos = path.pointAtPercent(1.0)
        
        # Calculate angle from arrow_pos to end_pos
        line = QLineF(arrow_pos, end_pos)
        if line.length() == 0:
            return
        
        angle = -line.angle()  # Qt angles are counter-clockwise
        
        # Create arrow polygon
        arrow_size = self._arrow_size + self._get_width() // 2
        
        # Arrow points
        p1 = QPointF(0, 0)
        p2 = QPointF(-arrow_size, arrow_size * 0.5)
        p3 = QPointF(-arrow_size, -arrow_size * 0.5)
        
        # Rotate and translate
        def rotate_point(p: QPointF, angle_deg: float) -> QPointF:
            rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            return QPointF(
                p.x() * cos_a - p.y() * sin_a,
                p.x() * sin_a + p.y() * cos_a
            )
        
        p1 = end_pos + rotate_point(p1, angle)
        p2 = end_pos + rotate_point(p2, angle)
        p3 = end_pos + rotate_point(p3, angle)
        
        arrow = QPolygonF([p1, p2, p3])
        
        # Draw filled arrow
        painter.setBrush(QBrush(self._get_color()))
        painter.setPen(QPen(Qt.transparent))
        painter.drawPolygon(arrow)
    
    def _draw_lock_icon(self, painter) -> None:
        """Draw a small lock icon in the middle of the link."""
        path = self.path()
        if path.isEmpty():
            return
        
        mid = path.pointAtPercent(0.5)
        size = 8
        
        # Draw lock body
        painter.setBrush(QBrush(QColor(80, 100, 130)))
        painter.setPen(QPen(QColor(60, 80, 110), 1))
        painter.drawRoundedRect(
            int(mid.x() - size/2), int(mid.y() - size/4),
            size, int(size * 0.75), 2, 2
        )
        
        # Draw lock shackle (arc)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(60, 80, 110), 2))
        painter.drawArc(
            int(mid.x() - size/3), int(mid.y() - size),
            int(size * 0.66), int(size * 0.75),
            0, 180 * 16  # Qt uses 1/16th degree units
        )
    
    def hoverEnterEvent(self, event) -> None:
        """Highlight on hover."""
        if not self._is_temporary:
            pen = self.pen()
            pen.setWidth(pen.width() + 2)
            self.setPen(pen)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event) -> None:
        """Remove highlight on hover exit."""
        self._update_pen()
        super().hoverLeaveEvent(event)
    
    def get_info_text(self) -> str:
        """Get tooltip/info text for the link."""
        if not self.link:
            return f"Columns: {len(self.columns)}"
        
        lines = [
            f"📊 {self.link.source_node_name} → {self.link.target_node_name}",
            f"Columns: {self.link.enabled_columns}/{self.link.total_columns}",
            f"Type: {self.link.data_type}",
            f"State: {self.link.state.value}",
        ]
        
        if self.link.estimated_memory_cost != "Unknown":
            lines.append(f"Memory: {self.link.estimated_memory_cost}")
        
        if self.link.error_message:
            lines.append(f"❌ {self.link.error_message}")
        elif self.link.warning_message:
            lines.append(f"⚠️ {self.link.warning_message}")
        
        return "\n".join(lines)
