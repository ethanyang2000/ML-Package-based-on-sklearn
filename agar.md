# agar 
1. modules
   1. GameAbstractions.py
      - class bound:
        - members: range of this area & height & width
        - methods: show & judge
      - class QuadItem:
        - members: cell & bound
      - class color:
        - rgb
      - class collision:
        - members: cell & check & d & p
   2. logger.py
      1. info & error
   3. QuadNode.py
      1. members:
         -  half width & half height
         -  parent
         -  level & maxlevel
         -  maxchildren
         -  bound(with cx & cy?)
         -  childNodes & items
      2. methods:
         1. insert: 
            - splited:
              - insert in quad
            - insert in items
            - item belongs to self

         2. remove:
         3. cleanup
         4. update: remove & insert
         5. clear
         6. contains
         7. find
         8. getQuad

   4. vec2: 2 vectors

2. entity
   1. cell.py: base class
      1. setRadius: radius & size & mass
      2. getAge
      3. onEat: recalculate size
      4. setBoost
      5. checkBorder
      - not implemented: 
        1. on Eaten
        2. onAdd
        3. onRemove
        4. getBound
        5. canEat
   2. food.py: type 1
      1. add & remove
   3. virus.py: type 2
      1. set characters & type
      2. canEat: celltype == 3?
      3. onEat: get larger or split?
      4. onEaten: explode the hunter
      5. explodeCell
      6. onAdd
      7. onRemove 
   4. playerCell.py: type 0
      - line 44 onEat?
      1. canEat
      2. getMoveR
      3. getSpeed
      4. onAdd
      5. onRemove
      - update reward: 
        1. onEat
        2. onEaten
   5. mothercell.py: type 2
      1. canEat
      2. onUpdate: generate food?
   6. ejectedMass.py: type 3
      1. onAdd & onRemove

3. players
   1. Player.py
      1. updateView: update viewbox
      2. step: given an action
         - comment at line 48?
         - line 55 press w?
      3. pressSpace: >2 split
      4. pressW: ejectMass
      5. setCenterPos
      6. getScale: view scale
      7. joinGame
      8. getViewBox: minx, maxx, miny, maxy
      9. in_view
      10. max/mincell
   2. Bot.py
      1. step: choose peace or aggressive randomly
         1. cooldown or repeat: action delay?
      2. peaceStep
      3. aggrassiveStep
4. agar_env
   1. step
      1. line 87 `actions[j][2] = 2`?
      2. call step_ for action_repeat times
      3. return obs, reward, done, info
   2. step_
      1. step agent & bot
      2. call parse_reward for each agent: mass & killed & kill
      3. call parse_obs...
   3. parse_obs:
      1. obs_id is a 578D array, first 560D is information of all entities around agent_id, last 28D is global information
      2. `n = [10, 5, 5, 10, 10]`  # , 10, 5, 5, 10, 10] # the agent can observe at most 10 self-cells, 5 foods, 5 virus, 10 other script agent cells, 10 other outside agent cells
      3. line 232 comment
      4. `s_size_i = [15, 7, 5, 15, 15]` information size
      5. call cell_obs, return cell type & feature
         1. type 0:
            - boostx
            - boosty
            - radius
            - log_radius
            - positionx
            - positiony
            - relativeposx
            - reposy
            - vx
            - vy
            - remerge
            - relative distance
            - maxsize?
            - minsize?
            - relative pos
         2. type 1:
            - radius
            - logradius
            - posx
            - posy
            - reposx
            - reposy
            - relative distance
         3. type 2:
            - posx
            - posy
            - reposx
            - reposy
            - relative distance
         4. type 3:
            - None
         5. global info
            - `obs_f[-1] = position_x`
            - `obs_f[-2] = position_y`
            - `obs_f[-3] = player.centerPos.sqDist() / self.server.config.r`
            - `obs_f[-4] = b_x #bound`
            - `obs_f[-5] = b_y`
            - `obs_f[-6] = len(obs[0])`
            - `obs_f[-7] = len(obs[1])`
            - `obs_f[-8] = len(obs[2])`
            - `obs_f[-9] = len(obs[3])`
            - `obs_f[-10] = len(obs[4])`
            - `obs_f[-11] = player.maxcell().radius / 400`
            - `obs_f[-12] = player.mincell().radius / 400`
            - `obs_f[-16:-13] = self.last_action[id * 3: id * 3 + 3]`
            - `obs_f[-17] = self.bot_speed`
            - `obs_f[-18] = (self.killed[id] != 0)`
            - `obs_f[-19] = (self.killed[1 - id] != 0)`
            - `obs_f[-20] = sum([c.mass for c in player.cells]) / 50`
            - `obs_f[-21] = sum([c.mass for c in self.agents[1 - id].cells]) / 50 `
   4. reset:
      - init server & players        
   