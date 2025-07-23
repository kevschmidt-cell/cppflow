import csv
from scipy.spatial.transform import Rotation as R

# Rotation um Z-Achse um 90 Grad
rot_90z = R.from_euler('z', 90, degrees=True)

input_file = 'cppflow/paths/hello_mini.csv'
output_file = 'cppflow/paths/hello_mini_rotated.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        t = row[0]
        x, y, z = map(float, row[1:4])
        qw, qx, qy, qz = map(float, row[4:8])

        # Position rotieren: 90° um Z
        x_new = -y
        y_new = x
        z_new = z

        # Quaternion rotieren (wichtig: scipy Rotation erwartet [x,y,z,w])
        quat_orig = R.from_quat([qx, qy, qz, qw])
        quat_new = rot_90z * quat_orig
        qx_new, qy_new, qz_new, qw_new = quat_new.as_quat()

        # Neue Werte in die Zeile schreiben
        row[1] = f"{x_new:.6f}"
        row[2] = f"{y_new:.6f}"
        row[3] = f"{z_new:.6f}"
        row[4] = f"{qw_new:.6f}"
        row[5] = f"{qx_new:.6f}"
        row[6] = f"{qy_new:.6f}"
        row[7] = f"{qz_new:.6f}"

        writer.writerow(row)
