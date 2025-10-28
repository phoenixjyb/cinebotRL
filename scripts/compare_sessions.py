"""
Compare Session 6 (frozen base) vs Session 7c (base movement)
"""

import numpy as np
from pathlib import Path

def load_session_stats(session_dir):
    """Load evaluation statistics from a session directory"""
    stats_file = session_dir / "evaluation_stats.npz"
    if stats_file.exists():
        data = np.load(stats_file)
        return {
            'episode_errors': data['episode_errors'],
            'episode_rewards': data['episode_rewards'],
            'base_movements': data['base_movements'],
            'all_ee_errors': data['all_ee_errors'],
        }
    return None

def compare_sessions():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Session 6: Frozen base (best previous result)
    session6_dir = PROJECT_ROOT / "logs" / "sb3" / "mobilemmtrackee_v0" / "20251023_105114"
    
    # Session 7c: Base movement with reachability guidance
    session7c_dir = PROJECT_ROOT / "logs" / "sb3" / "mobilemmtrackee_v0" / "20251027_180246"
    
    print(f"\n{'='*80}")
    print(f"SESSION COMPARISON: Session 6 (Frozen Base) vs Session 7c (Base Movement)")
    print(f"{'='*80}\n")
    
    # Load stats
    session6 = load_session_stats(session6_dir)
    session7c = load_session_stats(session7c_dir)
    
    if session6 is None:
        print("⚠️  Session 6 statistics not found. Run evaluation on Session 6 first.")
        print(f"   Expected at: {session6_dir / 'evaluation_stats.npz'}")
        
    if session7c is None:
        print("⚠️  Session 7c statistics not found. Run evaluation on Session 7c first.")
        print(f"   Expected at: {session7c_dir / 'evaluation_stats.npz'}")
    
    if session6 is None or session7c is None:
        print("\n💡 Run evaluation scripts first:")
        print("   python scripts/evaluate_session7c.py --save_stats")
        return
    
    # Compare tracking errors
    s6_mean = np.mean(session6['episode_errors'])
    s7c_mean = np.mean(session7c['episode_errors'])
    improvement = (s6_mean - s7c_mean) / s6_mean * 100
    
    print(f"📊 **TRACKING ERROR COMPARISON**\n")
    print(f"{'Metric':<25} {'Session 6':>15} {'Session 7c':>15} {'Change':>15}")
    print(f"{'-'*75}")
    print(f"{'Mean Error (m)':<25} {s6_mean:>15.4f} {s7c_mean:>15.4f} {improvement:>14.1f}%")
    print(f"{'Median Error (m)':<25} {np.median(session6['episode_errors']):>15.4f} "
          f"{np.median(session7c['episode_errors']):>15.4f} "
          f"{(np.median(session6['episode_errors']) - np.median(session7c['episode_errors'])) / np.median(session6['episode_errors']) * 100:>14.1f}%")
    print(f"{'Min Error (m)':<25} {np.min(session6['episode_errors']):>15.4f} "
          f"{np.min(session7c['episode_errors']):>15.4f}")
    print(f"{'Max Error (m)':<25} {np.max(session6['episode_errors']):>15.4f} "
          f"{np.max(session7c['episode_errors']):>15.4f}")
    print(f"{'Std Dev (m)':<25} {np.std(session6['episode_errors']):>15.4f} "
          f"{np.std(session7c['episode_errors']):>15.4f}\n")
    
    # Compare base movement
    s6_base = np.mean(session6['base_movements'])
    s7c_base = np.mean(session7c['base_movements'])
    
    print(f"🚗 **BASE MOVEMENT COMPARISON**\n")
    print(f"{'Metric':<25} {'Session 6':>15} {'Session 7c':>15} {'Change':>15}")
    print(f"{'-'*75}")
    print(f"{'Mean Movement (m)':<25} {s6_base:>15.4f} {s7c_base:>15.4f} "
          f"{'+' if s7c_base > s6_base else ''}{(s7c_base - s6_base):>14.4f}")
    print(f"{'Max Movement (m)':<25} {np.max(session6['base_movements']):>15.4f} "
          f"{np.max(session7c['base_movements']):>15.4f}")
    print(f"{'Median Movement (m)':<25} {np.median(session6['base_movements']):>15.4f} "
          f"{np.median(session7c['base_movements']):>15.4f}\n")
    
    # Performance classification comparison
    def classify_performance(errors):
        excellent = np.sum(errors < 0.1) / len(errors) * 100
        good = np.sum((errors >= 0.1) & (errors < 0.3)) / len(errors) * 100
        poor = np.sum((errors >= 0.3) & (errors < 2.0)) / len(errors) * 100
        broken = np.sum(errors >= 2.0) / len(errors) * 100
        return excellent, good, poor, broken
    
    s6_exc, s6_good, s6_poor, s6_broken = classify_performance(session6['episode_errors'])
    s7c_exc, s7c_good, s7c_poor, s7c_broken = classify_performance(session7c['episode_errors'])
    
    print(f"🎯 **PERFORMANCE CLASSIFICATION**\n")
    print(f"{'Category':<25} {'Session 6':>15} {'Session 7c':>15} {'Change':>15}")
    print(f"{'-'*75}")
    print(f"{'Excellent (<0.1m)':<25} {s6_exc:>14.1f}% {s7c_exc:>14.1f}% {s7c_exc - s6_exc:>+14.1f}%")
    print(f"{'Good (0.1-0.3m)':<25} {s6_good:>14.1f}% {s7c_good:>14.1f}% {s7c_good - s6_good:>+14.1f}%")
    print(f"{'Poor (0.3-2.0m)':<25} {s6_poor:>14.1f}% {s7c_poor:>14.1f}% {s7c_poor - s6_poor:>+14.1f}%")
    print(f"{'Broken (>2.0m)':<25} {s6_broken:>14.1f}% {s7c_broken:>14.1f}% {s7c_broken - s6_broken:>+14.1f}%\n")
    
    # Overall assessment
    print(f"{'='*80}")
    print(f"📈 **OVERALL ASSESSMENT**\n")
    
    if improvement > 0:
        print(f"✅ Session 7c (Base Movement) is **{improvement:.1f}% BETTER** than Session 6 (Frozen Base)")
    else:
        print(f"❌ Session 7c (Base Movement) is **{-improvement:.1f}% WORSE** than Session 6 (Frozen Base)")
    
    print(f"\n🔑 **KEY FINDINGS**:")
    print(f"   • Session 6: Base frozen, mean error {s6_mean:.4f}m")
    print(f"   • Session 7c: Base mobile ({s7c_base:.3f}m avg movement), mean error {s7c_mean:.4f}m")
    
    if improvement > 0:
        print(f"   • Base movement capability **IMPROVED** tracking accuracy!")
        print(f"   • System can now navigate to better vantage points")
    else:
        print(f"   • Base movement did not improve tracking (may need tuning)")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    compare_sessions()
