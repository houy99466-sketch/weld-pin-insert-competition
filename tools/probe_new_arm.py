# -*- coding: utf-8 -*-
"""只读探测：在指定串口上用达妙(DM)协议查询 1..8 号电机反馈。
不发送使能/运动指令，只 request_feedback + poll。
用法: python probe_new_arm.py COM22 [921600|1000000|4000000 ...]
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
from motorbridge.core import Controller

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM22'
BAUDS = [int(x) for x in sys.argv[2:]] or [921600, 1000000, 4000000, 3000000]
IDS = list(range(1, 9))
MODEL = '4310'

for baud in BAUDS:
    print('=' * 60)
    print(f'{PORT} @ {baud}  (DM 协议探测)')
    ctrl = None
    try:
        ctrl = Controller.from_dm_serial(PORT, baud)
    except Exception as e:
        print('  open failed:', e)
        continue
    try:
        motors = []
        for mid in IDS:
            try:
                motors.append((mid, ctrl.add_damiao_motor(mid, 0x10 + mid, MODEL)))
            except Exception as e:
                print(f'   id {mid}: add failed {e}')
        ok = 0
        for round_i in range(3):
            for _mid, m in motors:
                try:
                    m.request_feedback()
                except Exception:
                    pass
            for _ in range(4):
                try:
                    ctrl.poll_feedback_once()
                except Exception:
                    pass
                time.sleep(0.003)
        for mid, m in motors:
            try:
                s = m.get_state()
            except Exception as e:
                s = None
            if s is not None:
                ok += 1
                print(f'   id {mid:<2} pos={s.pos:+.4f} rad ({s.pos*57.29578:+8.2f}°) '
                      f'vel={s.vel:+.3f} torq={s.torq:+.3f} status={s.status_code} fault={s.fault_code}')
            else:
                print(f'   id {mid:<2} (无反馈)')
        print(f'  --> 有反馈电机数: {ok}/{len(IDS)}')
    finally:
        try:
            ctrl.close_bus()
        except Exception:
            pass
        try:
            ctrl.close()
        except Exception:
            pass
