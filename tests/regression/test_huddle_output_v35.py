"""Tight huddle output boundaries do not change upstream behavior detection."""
import copy

from mouse_behavior.visualization.behavior_packages import huddle_output_events


class Store:
    def __init__(self, rows):
        self.data = rows

    def rows(self, start, end):
        return iter(self.data[start:end])


def setup(length=130):
    rows = [dict(frame=f, detections=[dict(track_id=i, box=[i*9, 0, i*9+10, 10])
                                     for i in (1, 2, 3)]) for f in range(length)]
    event = dict(canonical_behavior='huddle', behavior='Huddling', mouse_ids=[1, 2, 3],
                 start_frame=0, end_frame=length, event_id='old',
                 canonical_support_frame_intervals=[[0, length]], canonical_support_frame_count=length)
    return rows, event, dict(source_video=dict(fps=10.)), dict(cm_per_pixel=.1)


def test_trim_arrival_and_departure_but_keep_full_seven_second_bout():
    rows, event, binding, scale = setup()
    for row in rows[:20]+rows[90:]:
        row['detections'][2]['box'] = [200, 0, 210, 10]
    before = copy.deepcopy((rows, event))
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert [(e['start_frame'], e['end_frame']) for e in output] == [(20, 90)]
    assert output[0]['canonical_support_frame_count'] == 70
    assert output[0]['boundary']['event_duration_s'] == 7
    assert (rows, event) == before


def test_visible_departure_splits_two_long_bouts_instead_of_filling_it():
    rows, event, binding, scale = setup(150)
    for row in rows[60:80]:
        row['detections'][2]['box'] = [200, 0, 210, 10]
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert [(e['start_frame'], e['end_frame']) for e in output] == [(0, 60), (80, 150)]


def test_short_tight_bout_not_padded_to_five_seconds():
    rows, event, binding, scale = setup()
    for row in rows[40:]:
        row['detections'][2]['box'] = [200, 0, 210, 10]
    output, audit = huddle_output_events([event], Store(rows), binding, scale)
    assert not output
    assert audit[0]['qualifying_sliding_windows'] == 0


def test_missing_member_cannot_supply_endpoints_or_five_seconds_support():
    rows, event, binding, scale = setup()
    for row in rows[:10]+rows[50:]:
        row['detections'] = row['detections'][:2]
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert not output


def test_internal_unknown_retained_without_creating_box_or_support():
    rows, event, binding, scale = setup(100)
    for row in rows[40:45]:
        row['detections'] = row['detections'][:2]
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert output[0]['canonical_support_frame_count'] == 100
    assert output[0]['observed_all_participants_frame_count'] == 95
    assert output[0]['inferred_or_partially_unobserved_frame_count'] == 5
    assert output[0]['canonical_support_frame_intervals'] == [[0, 100]]
    assert output[0]['observed_all_participants_frame_intervals'] == [[0, 40], [45, 100]]
    assert len(rows[42]['detections']) == 2


def test_other_behaviors_unmodified_and_duplicate_outputs_removed():
    rows, event, binding, scale = setup()
    other = dict(event, canonical_behavior='together', event_id='together_old')
    output, _ = huddle_output_events([event, copy.deepcopy(event), other], Store(rows), binding, scale)
    assert len(output) == 2
    assert output[1] == other


def test_sliding_window_accepts_supported_members_without_five_seconds_all_visible():
    rows, event, binding, scale = setup(80)
    for row in rows[1:-1]:
        missing = row['frame'] % 3 + 1
        row['detections'] = [d for d in row['detections'] if d['track_id'] != missing]
    before = copy.deepcopy(rows)
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert len(output) == 1
    assert (output[0]['start_frame'], output[0]['end_frame']) == (0, 80)
    assert output[0]['observed_all_participants_frame_count'] == 2
    assert output[0]['canonical_support_frame_count'] == 80
    assert rows == before


def test_short_pose_jitter_does_not_split_stable_sliding_process():
    rows, event, binding, scale = setup(80)
    rows[40]['detections'][2]['box'] = [200, 0, 210, 10]
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert len(output) == 1
    assert (output[0]['start_frame'], output[0]['end_frame']) == (0, 80)
    assert output[0]['canonical_support_frame_count'] == 79


def test_confirmed_departure_ends_at_first_separated_frame_not_confirmation_end():
    rows, event, binding, scale = setup(100)
    for row in rows[73:]:
        row['detections'][2]['box'] = [200, 0, 210, 10]
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert output[0]['end_frame'] == 73
    assert output[0]['boundary']['event_duration_s'] == 7.3


def test_rarely_observed_third_member_cannot_be_claimed_by_dense_pair():
    rows, event, binding, scale = setup(100)
    for row in rows[1:-1]:
        row['detections'] = row['detections'][:2]
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert not output


def test_sliding_window_does_not_export_spread_out_body_boxes_at_endpoints():
    rows, event, binding, scale = setup(130)
    for row in rows[:20]+rows[110:]:
        row['detections'][2]['box'] = [45, 0, 55, 10]
    output, _ = huddle_output_events([event], Store(rows), binding, scale)
    assert [(e['start_frame'], e['end_frame']) for e in output] == [(20, 110)]


def test_contained_member_subsets_are_not_exported_as_extra_review_clips():
    rows, event, binding, scale = setup(130)
    for row in rows:
        row['detections'].append(dict(track_id=4, box=[31, 0, 41, 10]))
    larger = dict(event, mouse_ids=[1, 2, 3, 4], event_id='larger')
    output, audit = huddle_output_events([event, larger], Store(rows), binding, scale)
    assert len(output) == 1 and output[0]['mouse_ids'] == [1, 2, 3, 4]
    assert any(row.get('reason') == 'contained_member_subset' for row in audit)
