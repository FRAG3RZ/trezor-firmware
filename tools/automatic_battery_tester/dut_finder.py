from dut.dut import Dut  

if __name__ == "__main__":

    cpu_ids = ["cpu_id_1"]  # Replace with actual CPU IDs

    for id in cpu_ids:
        try:
            dut = Dut(
                name=f"test_DUT_{id}",
                cpu_id=None,
                relay_port=1,
                relay_ctl=1,
                verbose=True,
            )
            break
        except Exception as e:
            print(f"Failed to find DUT {id}: {e}")

    

