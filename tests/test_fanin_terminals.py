"""Tests for M6 fan-in terminal support."""

import pytest
from recon_lite.graph import Graph, Node, NodeType, LinkType


def test_fanin_terminal_allows_multiple_parents():
    """A terminal (sensor) should allow multiple parent scripts."""
    g = Graph()
    
    # Create a shared sensor terminal
    sensor = Node("SharedSensor", NodeType.TERMINAL)
    g.add_node(sensor)
    
    # Create two scripts that both want to query the sensor
    script_a = Node("ScriptA", NodeType.SCRIPT)
    script_b = Node("ScriptB", NodeType.SCRIPT)
    g.add_node(script_a)
    g.add_node(script_b)
    
    # Both scripts should be able to SUB-link to the same terminal
    g.add_edge("ScriptA", "SharedSensor", LinkType.SUB)
    g.add_edge("ScriptB", "SharedSensor", LinkType.SUB)  # Should NOT raise
    
    # Verify both are tracked as parents
    assert g.all_parents("SharedSensor") == ["ScriptA", "ScriptB"]
    assert g.is_fanin_terminal("SharedSensor")


def test_script_single_parent_enforced():
    """A script node should only allow one parent."""
    g = Graph()
    
    # Create parent scripts
    parent_a = Node("ParentA", NodeType.SCRIPT)
    parent_b = Node("ParentB", NodeType.SCRIPT)
    child = Node("Child", NodeType.SCRIPT)
    # Need a terminal for the child to be valid
    terminal = Node("Terminal", NodeType.TERMINAL)
    
    g.add_node(parent_a)
    g.add_node(parent_b)
    g.add_node(child)
    g.add_node(terminal)
    
    # Add child's terminal first
    g.add_edge("Child", "Terminal", LinkType.SUB)
    
    # First parent should work
    g.add_edge("ParentA", "Child", LinkType.SUB)
    
    # Second parent should raise
    with pytest.raises(ValueError, match="already has a parent"):
        g.add_edge("ParentB", "Child", LinkType.SUB)


def test_terminal_sur_links():
    """Terminal should be able to originate SUR links back to parents."""
    g = Graph()
    
    sensor = Node("Sensor", NodeType.TERMINAL)
    script = Node("Script", NodeType.SCRIPT)
    
    g.add_node(sensor)
    g.add_node(script)
    
    # SUB from script to sensor
    g.add_edge("Script", "Sensor", LinkType.SUB)
    # SUR from sensor to script (confirmation path)
    g.add_edge("Sensor", "Script", LinkType.SUR)
    
    # Should work without error
    assert g.parent_of("Sensor") == "Script"


def test_fanin_terminal_primary_parent():
    """First parent should be the 'primary' for backward compatibility."""
    g = Graph()
    
    sensor = Node("Sensor", NodeType.TERMINAL)
    script_a = Node("ScriptA", NodeType.SCRIPT)
    script_b = Node("ScriptB", NodeType.SCRIPT)
    
    g.add_node(sensor)
    g.add_node(script_a)
    g.add_node(script_b)
    
    g.add_edge("ScriptA", "Sensor", LinkType.SUB)
    g.add_edge("ScriptB", "Sensor", LinkType.SUB)
    
    # First parent is primary
    assert g.parent_of("Sensor") == "ScriptA"
    # All parents accessible
    assert set(g.all_parents("Sensor")) == {"ScriptA", "ScriptB"}


def test_non_fanin_terminal():
    """Terminal with single parent should not be marked as fan-in."""
    g = Graph()
    
    sensor = Node("Sensor", NodeType.TERMINAL)
    script = Node("Script", NodeType.SCRIPT)
    
    g.add_node(sensor)
    g.add_node(script)
    
    g.add_edge("Script", "Sensor", LinkType.SUB)
    
    assert not g.is_fanin_terminal("Sensor")
    assert g.all_parents("Sensor") == ["Script"]

