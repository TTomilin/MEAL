#!/usr/bin/env python3
"""
Demonstration of complementary restrictions feature in IPPO_CL.py

This script shows how to enable and use the complementary restrictions feature
where one agent cannot pick up onions while the other cannot pick up plates.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from baselines.IPPO_CL import Config

def demo_complementary_restrictions():
    """Demonstrate how to use complementary restrictions"""
    
    print("=" * 60)
    print("COMPLEMENTARY RESTRICTIONS DEMO")
    print("=" * 60)
    
    print("\n1. Creating config with complementary restrictions DISABLED (default):")
    config_disabled = Config()
    print(f"   complementary_restrictions = {config_disabled.complementary_restrictions}")
    print("   → Agents can pick up any objects (normal behavior)")
    
    print("\n2. Creating config with complementary restrictions ENABLED:")
    config_enabled = Config()
    config_enabled.complementary_restrictions = True
    print(f"   complementary_restrictions = {config_enabled.complementary_restrictions}")
    print("   → One agent cannot pick onions, other cannot pick plates")
    print("   → Roles are randomly assigned at the start of each task")
    
    print("\n3. How to use in command line:")
    print("   python baselines/IPPO_CL.py --complementary_restrictions=True")
    
    print("\n4. How it works:")
    print("   • At the start of each task, roles are randomly assigned:")
    print("     - Role 0: Agent 0 cannot pick onions, Agent 1 cannot pick plates")
    print("     - Role 1: Agent 0 cannot pick plates, Agent 1 cannot pick onions")
    print("   • Roles stay the same throughout the entire task")
    print("   • New roles are assigned for the next task")
    
    print("\n5. Example task sequence:")
    print("   Task 0: Agent 0 cannot pick onions, Agent 1 cannot pick plates")
    print("   Task 1: Agent 0 cannot pick plates, Agent 1 cannot pick onions")
    print("   Task 2: Agent 0 cannot pick onions, Agent 1 cannot pick plates")
    print("   (Random assignment for each task)")
    
    print("\n6. Benefits:")
    print("   • Forces agents to specialize and cooperate")
    print("   • Creates more challenging scenarios")
    print("   • Tests agent adaptability to different roles")
    print("   • Prevents one agent from doing everything")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    demo_complementary_restrictions()