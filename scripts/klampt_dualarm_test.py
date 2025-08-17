from klampt import WorldModel
import numpy as np

def main():
    world = WorldModel()

    # URDF-Dateipfade anpassen!
    urdf_left = "path/to/left_arm.urdf"
    urdf_right = "path/to/right_arm.urdf"
    urdf_object = "path/to/object.urdf"

    assert world.readFile(urdf_left), "Linker Arm konnte nicht geladen werden"
    assert world.readFile(urdf_right), "Rechter Arm konnte nicht geladen werden"
    assert world.readFile(urdf_object), "Objekt konnte nicht geladen werden"

    robot_left = world.robot(0)
    robot_right = world.robot(1)
    obj_robot = world.robot(2)

    # Beispiel-Gelenkwinkel (replace mit deinen)
    q_left = [0.0]*robot_left.numLinks()
    q_right = [0.0]*robot_right.numLinks()

    # Beispiel-Objekt-Transform (Identität = keine Verschiebung)
    T_obj = np.eye(4)

    def collision_check():
        if robot_left.selfCollision():
            print("Linker Arm Self-Collision!")
            return True
        if robot_right.selfCollision():
            print("Rechter Arm Self-Collision!")
            return True
        if robot_left.collides(robot_right):
            print("Arme kollidieren miteinander!")
            return True
        if robot_left.collides(obj_robot):
            print("Linker Arm kollidiert mit Objekt!")
            return True
        if robot_right.collides(obj_robot):
            print("Rechter Arm kollidiert mit Objekt!")
            return True
        return False

    # Konfigurationen setzen
    robot_left.setConfig(q_left)
    robot_right.setConfig(q_right)
    obj_robot.setLinkTransform(0, T_obj)

    # Kollisionsprüfung
    if collision_check():
        print("Kollision erkannt!")
    else:
        print("Keine Kollision!")

if __name__ == "__main__":
    main()
